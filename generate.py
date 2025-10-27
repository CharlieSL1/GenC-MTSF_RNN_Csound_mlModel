import torch
import torch.nn as nn
import soundfile as sf
import ctcsound
import json
import os
from pathlib import Path
import pandas as pd

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def create_mtsf_model(input_size, hidden_size, output_size):
    """Create MTSF model (same as in train.py)"""
    class MTSFModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.forecast_head = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            rnn_out, hidden = self.rnn(x)
            last_hidden = rnn_out[:, -1, :]
            forecast = self.forecast_head(last_hidden)
            return forecast, hidden
    
    return MTSFModel(input_size, hidden_size, output_size)

def load_trained_model():
    """Load the trained model"""
    model_path = Path("mtsf_model.pt")
    if not model_path.exists():
        print("Error: mtsf_model.pt not found. Please train the model first.")
        return None, None, None
    
    # Load metadata to get parameter info
    metadata_path = Path("Dataset/dataset_metadata.csv")
    if not metadata_path.exists():
        print("Error: Dataset metadata not found.")
        return None, None, None
    
    df = pd.read_csv(metadata_path)
    
    # Get parameter names from first row
    first_params = json.loads(df.iloc[0]['parameters'].replace("'", '"'))
    param_names = [key for key in first_params.keys() 
                   if key not in ['form', 'filebutton32', 'filebutton33', 'cabbageJSONData']]
    
    # Load audio to determine input size
    first_audio = Path(df.iloc[0]['output_file'])
    if first_audio.exists():
        audio_data, _ = sf.read(str(first_audio))
        input_size = 1  # Mono audio
        hidden_size = 64
        output_size = len(param_names)
        
        # Create and load model
        model = create_mtsf_model(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        return model, param_names, len(audio_data)
    
    return None, None, None

def predict_parameters(model, audio_file_path, sequence_length):
    """Predict parameters for a given audio file"""
    try:
        # Load audio file
        audio_data, sample_rate = sf.read(str(audio_file_path))
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        
        # Pad or truncate to match training sequence length
        if len(audio_tensor) < sequence_length:
            audio_tensor = torch.cat([audio_tensor, torch.zeros(sequence_length - len(audio_tensor))])
        elif len(audio_tensor) > sequence_length:
            audio_tensor = audio_tensor[:sequence_length]
        
        # Add batch and channel dimensions
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        
        # Move to device
        audio_tensor = audio_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            forecast, _ = model(audio_tensor)
            return forecast.squeeze().cpu().numpy()  # Move back to CPU for numpy conversion
    
    except Exception as e:
        print(f"Error predicting parameters for {audio_file_path}: {e}")
        return None

def generate_sound_with_csound(input_audio_path, predicted_params, param_names, output_path):
    """Generate new sound using Csound with predicted parameters"""
    try:
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Find a CSD file to use as template
        csound_cab_path = Path("CsoundCAB")
        csd_files = list(csound_cab_path.glob("**/*.csd"))
        
        if not csd_files:
            print("No CSD files found in CsoundCAB folder")
            return None
        
        # Use the first CSD file as template
        template_csd = csd_files[0]
        print(f"Using template CSD: {template_csd}")
        
        # Create output filename
        input_name = Path(input_audio_path).stem
        output_filename = f"generated_{input_name}.wav"
        output_filepath = output_dir / output_filename
        
        # Initialize Csound
        cs = ctcsound.Csound()
        cs.setOption("-W")        # WAV output
        cs.setOption("-f")        # 32-bit float
        cs.setOption("-r44100")   # Sample rate 44.1kHz
        cs.setOption("-o" + str(output_filepath))  # Output file
        cs.setOption("-d")        # Disable display
        cs.setOption("-t10")      # Duration 10 seconds
        
        # Compile CSD file
        result = cs.compileCsd(str(template_csd))
        if result != 0:
            print(f"Error compiling CSD file: {result}")
            cs.cleanup()
            return None
        
        # Set predicted parameters
        for i, param_name in enumerate(param_names):
            if i < len(predicted_params):
                cs.setControlChannel(param_name, float(predicted_params[i]))
                print(f"  Set {param_name} = {predicted_params[i]:.4f}")
        
        # Perform synthesis
        cs.start()
        cs.perform()
        cs.stop()
        cs.cleanup()
        
        print(f"Generated: {output_filename}")
        return str(output_filepath)
        
    except Exception as e:
        print(f"Error generating sound: {e}")
        return None

def generate_parameters(input_audio_path):
    """Generate new parameters from input audio - returns parameter values"""
    print(f"Generating parameters from: {input_audio_path}")
    
    # Load trained model
    model, param_names, sequence_length = load_trained_model()
    if model is None:
        return None
    
    print(f"Loaded model with {len(param_names)} parameters")
    print(f"Parameter names: {param_names}")
    
    # Predict parameters
    predicted_params = predict_parameters(model, input_audio_path, sequence_length)
    if predicted_params is None:
        return None
    
    # Create parameter dictionary
    param_dict = {}
    for i, param_name in enumerate(param_names):
        if i < len(predicted_params):
            param_dict[param_name] = float(predicted_params[i])
    
    print(f"Generated parameters:")
    for param_name, value in param_dict.items():
        print(f"  {param_name}: {value:.4f}")
    
    return param_dict

def create_csound_with_parameters(param_dict, template_csd_path, output_name):
    """Use Csound to generate new sound and CSD file with given parameters"""
    try:
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Create output filenames
        wav_filename = f"{output_name}.wav"
        csd_filename = f"{output_name}.csd"
        wav_filepath = output_dir / wav_filename
        csd_filepath = output_dir / csd_filename
        
        # Initialize Csound
        cs = ctcsound.Csound()
        cs.setOption("-W")        # WAV output
        cs.setOption("-f")        # 32-bit float
        cs.setOption("-r44100")   # Sample rate 44.1kHz
        cs.setOption("-o" + str(wav_filepath))  # Output file
        cs.setOption("-d")        # Disable display
        cs.setOption("-t10")      # Duration 10 seconds
        
        # Compile CSD file
        result = cs.compileCsd(str(template_csd_path))
        if result != 0:
            print(f"Error compiling CSD file: {result}")
            cs.cleanup()
            return None, None
        
        # Set parameters
        for param_name, value in param_dict.items():
            cs.setControlChannel(param_name, value)
            print(f"  Set {param_name} = {value:.4f}")
        
        # Perform synthesis
        cs.start()
        cs.perform()
        cs.stop()
        cs.cleanup()
        
        # Create new CSD file with the parameters
        create_csd_file_with_parameters(template_csd_path, csd_filepath, param_dict)
        
        print(f"Generated: {wav_filename} and {csd_filename}")
        return str(wav_filepath), str(csd_filepath)
        
    except Exception as e:
        print(f"Error generating sound with Csound: {e}")
        return None, None

def create_csd_file_with_parameters(template_csd_path, output_csd_path, param_dict):
    """Create a new CSD file with embedded parameters"""
    try:
        # Read template CSD file
        with open(template_csd_path, 'r') as f:
            csd_content = f.read()
        
        # Add parameter initialization at the beginning
        param_init = "\n; Generated Parameters\n"
        for param_name, value in param_dict.items():
            param_init += f"gk{param_name} init {value:.6f}\n"
        
        # Insert parameters after the first line (usually <CsoundSynthesizer>)
        lines = csd_content.split('\n')
        if len(lines) > 1:
            lines.insert(1, param_init)
            new_content = '\n'.join(lines)
        else:
            new_content = csd_content + param_init
        
        # Write new CSD file
        with open(output_csd_path, 'w') as f:
            f.write(new_content)
        
        print(f"Created CSD file: {output_csd_path}")
        
    except Exception as e:
        print(f"Error creating CSD file: {e}")

def generate_from_dataset_samples():
    """Generate new sounds using existing dataset samples as input"""
    dataset_path = Path("Dataset")
    if not dataset_path.exists():
        print("Dataset folder not found. Please run Csound.py first.")
        return
    
    # Get list of WAV files in dataset
    wav_files = list(dataset_path.glob("*.wav"))
    if not wav_files:
        print("No WAV files found in Dataset folder.")
        return
    
    print(f"Found {len(wav_files)} audio files in dataset")
    
    # Generate new sounds for first few samples
    for i, wav_file in enumerate(wav_files[:3]):  # Process first 3 files
        print(f"\n--- Processing {i+1}/{min(3, len(wav_files))}: {wav_file.name} ---")
        generate_from_audio(str(wav_file))

def complete_workflow(input_audio_path):
    """Complete workflow: Audio -> Parameters -> Csound -> WAV + CSD"""
    print("=== Complete Workflow ===")
    print("1. Generate parameters from audio")
    print("2. Use parameters with Csound to create new sound")
    print("3. Export WAV file and CSD file")
    print()
    
    # Step 1: Generate parameters
    param_dict = generate_parameters(input_audio_path)
    if param_dict is None:
        print("Failed to generate parameters")
        return None, None
    
    # Step 2: Find template CSD file
    csound_cab_path = Path("CsoundCAB")
    csd_files = list(csound_cab_path.glob("**/*.csd"))
    
    if not csd_files:
        print("No CSD files found in CsoundCAB folder")
        return None, None
    
    template_csd = csd_files[0]
    print(f"Using template CSD: {template_csd}")
    
    # Step 3: Generate new sound and CSD with parameters
    input_name = Path(input_audio_path).stem
    wav_file, csd_file = create_csound_with_parameters(param_dict, template_csd, f"generated_{input_name}")
    
    if wav_file and csd_file:
        print(f"\n=== Workflow Complete ===")
        print(f"Generated WAV: {wav_file}")
        print(f"Generated CSD: {csd_file}")
        return wav_file, csd_file
    else:
        print("Failed to complete workflow")
        return None, None

if __name__ == "__main__":
    print("=== Csound Parameter Generator ===")
    print("This script generates new sounds using trained MTSF model")
    print()
    
    # Check if model exists
    if not Path("mtsf_model.pt").exists():
        print("Error: mtsf_model.pt not found.")
        print("Please run train.py first to train the model.")
        exit(1)
    
    # Demo: Complete workflow with dataset samples
    dataset_path = Path("Dataset")
    if dataset_path.exists():
        wav_files = list(dataset_path.glob("*.wav"))
        if wav_files:
            print(f"Running complete workflow on: {wav_files[0].name}")
            complete_workflow(str(wav_files[0]))
        else:
            print("No WAV files found in Dataset folder.")
    else:
        print("Dataset folder not found. Please run Csound.py first.")
    
    print("\n=== Generation Complete ===")
    print("Check the 'output' folder for generated sounds!")
