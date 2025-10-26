# -*- coding: utf-8 -*-
import ctcsound
import json
from pathlib import Path
import pandas as pd
import soundfile as sf
import numpy as np
import time
import sys
import os
from datetime import datetime

TARGET_SECONDS = 10.0
SR = 44100
RMS_SILENCE_THRESHOLD = 1e-3     # 全局RMS阈值：纯静音判定
ACTIVE_ABS_THRESHOLD = 1e-3      # 样本绝对值阈值：活动样本判定
MIN_ACTIVE_RATIO = 0.02          # 至少2%的样本“有声”，否则判为静音
FADE_MS = 5.0                    # 渐入/渐出毫秒，避免点击

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def fade_in_out(x: np.ndarray, sr: int):
    n = x.shape[0]
    fade_len = max(1, int(sr * (FADE_MS / 1000.0)))
    if fade_len*2 >= n:
        return x
    ramp_in = np.linspace(0.0, 1.0, fade_len, dtype=x.dtype)
    ramp_out = np.linspace(1.0, 0.0, fade_len, dtype=x.dtype)
    if x.ndim == 1:
        x[:fade_len] *= ramp_in
        x[-fade_len:] *= ramp_out
    else:
        x[:fade_len, :] = (x[:fade_len, :].T * ramp_in).T
        x[-fade_len:, :] = (x[-fade_len:, :].T * ramp_out).T
    return x

def normalize_peak(x: np.ndarray, peak_dbfs=-1.0):
    # 目标峰值：-1 dBFS
    target = 10 ** (peak_dbfs / 20.0)
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak > 0:
        g = target / peak
        if g < 1.0:
            x = x * g
    return x

def is_silent(x: np.ndarray):
    if x.size == 0:
        return True
    # 转单声道评估
    mono = x if x.ndim == 1 else x.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
    active_ratio = float(np.mean(np.abs(mono) > ACTIVE_ABS_THRESHOLD)) if mono.size else 0.0
    return rms < RMS_SILENCE_THRESHOLD or active_ratio < MIN_ACTIVE_RATIO, rms, active_ratio

def to_exact_10s_non_silent(x: np.ndarray, sr: int):
    """确保结果严格10秒；不足则循环拼接以避免补零静音，超出则截断。"""
    samples_target = int(sr * TARGET_SECONDS)
    if x.ndim == 1:
        x_work = x
        while x_work.shape[0] < samples_target and x.size > 0:
            x_work = np.concatenate([x_work, x], axis=0)
        x_work = x_work[:samples_target]
    else:
        x_work = x
        while x_work.shape[0] < samples_target and x.shape[0] > 0:
            x_work = np.concatenate([x_work, x], axis=0)
        x_work = x_work[:samples_target, :]
    # 渐入/渐出 + 峰值归一到 -1 dBFS
    x_work = fade_in_out(x_work, sr)
    x_work = normalize_peak(x_work, -1.0)
    return x_work

def safe_json_write_line(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def process_single_preset_ctcsound(csd_file: Path, preset_name: str, parameters: dict, output_path: Path):
    """
    使用 performKsmps 精确渲染10秒，且跨平台稳定。
    """
    output_filename = f"{csd_file.stem}_{preset_name}.wav"
    output_filepath = output_path / output_filename
    print(f"    ▶ {preset_name}")

    # 1) 初始化 Csound & 选项
    cs = ctcsound.Csound()
    cs.setOption("-W")                     # WAV
    cs.setOption("-f")                     # 32-bit float
    cs.setOption(f"-r{SR}")                # 采样率
    cs.setOption("-d")                     # 禁止显示
    cs.setOption("-n")                     # 不写临时声卡
    cs.setOption("-z")                     # 关闭显示窗口（等价“无GUI”）
    cs.setOption("-m0")                    # 减少控制台噪声
    cs.setOption("-J")                     # 禁止实时MIDI(更稳)
    cs.setOption("-K")                     # 禁止实时音频输入
    cs.setOption("-b1024")                 # 小buffer更稳定
    cs.setOption("-B2048")
    cs.setOption("-g")                     # 快速启动
    cs.setOption("-o" + str(output_filepath))

    # 2) 编译 CSD
    res = cs.compileCsd(str(csd_file))
    if res != 0:
        print(f"      ✖ 编译失败: {res}")
        cs.cleanup()
        return None

    # 3) 设置参数（强制 dur=10 覆盖）
    for k, v in parameters.items():
        if k in ['form','filebutton32','filebutton33','cabbageJSONData']:
            continue
        try:
            cs.setControlChannel(k, float(v))
        except Exception as e:
            print(f"      ⚠ 参数 {k}={v} 设置失败: {e}")
    # 强制持续时间通道（如果补丁里有用到）
    try:
        cs.setControlChannel("dur", float(TARGET_SECONDS))
    except Exception:
        pass

    # 4) 启动并以 Ksmps 为步长渲染固定时长
    start = time.monotonic()
    try:
        cs.start()
        sr = int(cs.GetSr()) if hasattr(cs, "GetSr") else SR
        ksmps = int(cs.GetKsmps()) if hasattr(cs, "GetKsmps") else 64
        samples_target = int(TARGET_SECONDS * sr)
        samples_done = 0
        hard_deadline = start + 30.0   # 单预设硬超时 30s（墙钟）

        while samples_done < samples_target:
            if time.monotonic() > hard_deadline:
                print("      ⏱ 超时，强制停止")
                break
            # performKsmps 在乐谱结束会返回非0，我们继续循环直到满10秒（仍会写入“空样本”）
            ret = cs.performKsmps()
            samples_done += ksmps
            if ret != 0:
                # 乐谱已结束，但我们继续推进直到满10秒，避免卡住
                pass
    except Exception as e:
        print(f"      ✖ 渲染错误: {e}")
    finally:
        try:
            cs.stop()
        except Exception:
            pass
        try:
            cs.cleanup()
        except Exception:
            pass

    # 5) 校验 & 后处理为严格 10s 且非纯静音
    if not output_filepath.exists():
        print("      ✖ 未生成文件")
        return None

    try:
        data, sample_rate = sf.read(str(output_filepath), always_2d=False)
    except Exception as e:
        print(f"      ✖ 读取失败: {e}")
        try:
            output_filepath.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    if data.size == 0:
        print("      ✖ 空文件，删除")
        output_filepath.unlink(missing_ok=True)
        return None

    # 记录原始时长
    orig_dur = float(data.shape[0]) / float(sample_rate)

    # 静音检测
    silent, rms_val, active_ratio = is_silent(data)
    if silent:
        print(f"      ✖ 纯静音 (RMS={rms_val:.6f}, 活动占比={active_ratio:.3f})，删除")
        output_filepath.unlink(missing_ok=True)
        return None

    # 重整为严格10秒（循环拼接/截断 + 渐入/出 + 峰值限制）
    data_10s = to_exact_10s_non_silent(data, sample_rate)

    # 再次静音防御
    silent2, rms_val2, active_ratio2 = is_silent(data_10s)
    if silent2:
        print(f"      ✖ 重整后仍静音 (RMS={rms_val2:.6f}, 活动占比={active_ratio2:.3f})，删除")
        output_filepath.unlink(missing_ok=True)
        return None

    # 覆写到磁盘（float32）
    sf.write(str(output_filepath), data_10s.astype(np.float32), sample_rate, subtype="FLOAT")
    print(f"      ✓ 生成 {output_filepath.name}  ({orig_dur:.2f}s → 10.00s, RMS={rms_val2:.6f}, 活动={active_ratio2:.3f})")

    # 返回用于汇总
    applied_params = dict(parameters)
    applied_params["dur"] = TARGET_SECONDS  # 明确记录导出时强制覆盖的 dur
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sound_name": csd_file.stem,
        "preset_name": preset_name,
        "output_file": str(output_filepath),
        "sr": sample_rate,
        "metrics": {
            "orig_duration_sec": orig_dur,
            "final_duration_sec": TARGET_SECONDS,
            "rms": rms_val2,
            "active_ratio": active_ratio2
        },
        "parameters": applied_params
    }

def cleanup_dataset():
    dataset_path = Path("Dataset")
    if not dataset_path.exists():
        return
    print("清理异常文件...")
    cleaned = 0
    for f in dataset_path.glob("*.wav"):
        try:
            if f.stat().st_size < 1024 or f.stat().st_size > 100 * 1024 * 1024:
                print(f"  删除异常大小: {f.name}")
                f.unlink()
                cleaned += 1
                continue
            try:
                d, _ = sf.read(str(f))
                if d.size == 0:
                    print(f"  删除空音频: {f.name}")
                    f.unlink(); cleaned += 1
            except Exception:
                print(f"  删除损坏音频: {f.name}")
                f.unlink(); cleaned += 1
        except Exception as e:
            print(f"  检查失败 {f.name}: {e}")
    print(f"清理完成，移除 {cleaned} 个文件")

def process_csd_files():
    cleanup_dataset()

    csound_cab_path = Path("CsoundCAB")
    output_path = Path("Dataset")
    ensure_dir(output_path)

    # 参数日志（逐行JSON）
    params_log_path = output_path / "parameters_log.jsonl"
    # 元数据CSV
    meta_csv_path = output_path / "dataset_metadata.csv"

    csd_files = list(csound_cab_path.glob("**/*.csd"))
    print(f"发现 {len(csd_files)} 个 CSD")

    processed_rows = []
    total_presets = 0

    # 预统计
    for csd in csd_files:
        snap_file = csd.with_suffix(".snaps")
        if snap_file.exists():
            try:
                snap_data = json.load(open(snap_file, "r", encoding="utf-8"))
                total_presets += len(snap_data)
            except Exception:
                pass
    print(f"预设总数: {total_presets}")

    done = 0
    for csd in csd_files:
        print(f"\n处理：{csd}")
        snap_file = csd.with_suffix(".snaps")
        if not snap_file.exists():
            print("  无 .snaps，跳过")
            continue
        try:
            snap_data = json.load(open(snap_file, "r", encoding="utf-8"))
        except Exception as e:
            print(f"  读取 .snaps 失败: {e}")
            continue

        for preset_name, parameters in snap_data.items():
            res = process_single_preset_ctcsound(csd, preset_name, parameters, output_path)
            done += 1
            print(f"  进度: {done}/{total_presets}")

            if res:
                # 追加到 JSONL
                try:
                    safe_json_write_line(params_log_path, res)
                except Exception as e:
                    print(f"  ⚠ 写参数日志失败: {e}")

                # 收集CSV行
                processed_rows.append({
                    "timestamp": res["timestamp"],
                    "sound_name": res["sound_name"],
                    "preset_name": res["preset_name"],
                    "output_file": res["output_file"],
                    "sr": res["sr"],
                    "orig_duration_sec": res["metrics"]["orig_duration_sec"],
                    "final_duration_sec": res["metrics"]["final_duration_sec"],
                    "rms": res["metrics"]["rms"],
                    "active_ratio": res["metrics"]["active_ratio"]
                })

            # 每处理5条保存一次CSV
            if len(processed_rows) and (len(processed_rows) % 5 == 0):
                try:
                    pd.DataFrame(processed_rows).to_csv(meta_csv_path, index=False)
                    print(f"  已保存CSV进度（{len(processed_rows)} 条）")
                except Exception as e:
                    print(f"  ⚠ 保存CSV失败: {e}")

            # 小憩，避免文件系统抖动
            time.sleep(0.05)

    cleanup_dataset()

    if processed_rows:
        try:
            pd.DataFrame(processed_rows).to_csv(meta_csv_path, index=False)
            print(f"\n完成！共生成 {len(processed_rows)} 个有效音频")
            print(f"CSV: {meta_csv_path}")
            print(f"参数日志(JSONL): {params_log_path}")
        except Exception as e:
            print(f"最终保存CSV失败: {e}")
    else:
        print("\n无有效音频生成（可能全部为静音/异常）")

    return processed_rows

if __name__ == "__main__":
    process_csd_files()
