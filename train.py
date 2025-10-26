import torch 
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
import json

def load_dataset():
    """Load audio files and parameters from Dataset folder"""
    dataset_path = Path("Dataset")
    
    # Load parameters from JSONL file
    params_log_file = dataset_path / "parameters_log.jsonl"
    if not params_log_file.exists():
        print("No parameters_log.jsonl found. Please run Csound.py first to generate dataset.")
        return None, None
    
    # Load metadata CSV for file paths
    metadata_file = dataset_path / "dataset_metadata.csv"
    if not metadata_file.exists():
        print("No dataset_metadata.csv found. Please run Csound.py first to generate dataset.")
        return None, None
    
    df = pd.read_csv(metadata_file)
    print(f"Loaded metadata for {len(df)} files")
    
    X_data = []  # Audio features
    y_data = []  # Parameters
    parameter_names = None  # Will be set from first valid entry
    
    # Load parameters from JSONL
    params_data = []
    with open(params_log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                params_data.append(json.loads(line.strip()))
            except Exception as e:
                print(f"Error parsing JSON line: {e}")
                continue
    
    print(f"Loaded {len(params_data)} parameter entries")
    
    # Create a mapping from output_file to parameters
    params_map = {}
    for entry in params_data:
        output_file = entry['output_file']
        params_map[output_file] = entry['parameters']
    
    for _, row in df.iterrows():
        audio_file = Path(row['output_file'])
        if audio_file.exists():
            try:
                # Load audio file
                audio_data, sample_rate = sf.read(str(audio_file))
                
                # Convert to tensor and add to X_data
                audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
                X_data.append(audio_tensor)
                
                # Get parameters from the mapping
                output_file_str = str(audio_file)
                if output_file_str in params_map:
                    params = params_map[output_file_str]
                    param_values = []
                    
                    # Set parameter names from first entry
                    if parameter_names is None:
                        parameter_names = []
                        for key, value in params.items():
                            if key not in ['form', 'filebutton32', 'filebutton33', 'cabbageJSONData', 'dur']:
                                parameter_names.append(key)
                    
                    # Extract parameter values in consistent order
                    for key in parameter_names:
                        if key in params:
                            param_values.append(float(params[key]))
                        else:
                            param_values.append(0.0)  # Default value for missing parameters
                    
                    y_data.append(torch.tensor(param_values, dtype=torch.float32))
                else:
                    print(f"No parameters found for {audio_file}")
                    continue
                
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
                continue
    
    if not X_data:
        print("No valid audio files found!")
        return None, None
    
    print(f"Parameter names: {parameter_names}")
    print(f"Number of parameters: {len(parameter_names) if parameter_names else 0}")
    
    # All audio files should be exactly 10 seconds (441000 samples at 44.1kHz)
    # No need to pad since they're all the same length
    X = torch.stack(X_data)  # Stack the audio tensors
    y = torch.stack(y_data)
    
    # Handle stereo audio - convert to mono or keep stereo properly
    if X.dim() == 4:  # [batch, time, channels, extra_dim]
        X = X.squeeze(-1)  # Remove extra dimension
    if X.dim() == 3 and X.shape[-1] == 2:  # [batch, time, 2] - stereo
        X = X.mean(dim=-1, keepdim=True)  # Convert to mono: [batch, time, 1]
    elif X.dim() == 2:  # [batch, time] - mono
        X = X.unsqueeze(-1)  # Add channel dimension: [batch, time, 1]
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Audio duration: {X.shape[1] / 44100:.2f} seconds per file")
    return X, y

def create_mtsf_model(input_size, hidden_size, output_size):
    """Create MTSF (Multivariate Time Series Forecasting) model"""
    class MTSFModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.forecast_head = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            # x shape: (batch, sequence_length, input_size)
            rnn_out, hidden = self.rnn(x)
            # Use last hidden state for forecasting
            last_hidden = rnn_out[:, -1, :]  # (batch, hidden_size)
            forecast = self.forecast_head(last_hidden)
            return forecast, hidden
    
    return MTSFModel(input_size, hidden_size, output_size)

# Load dataset
print("Loading dataset...")
X, y = load_dataset()

if X is not None and y is not None:
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create data loaders
    batch_size = 4
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Model parameters
    input_size = X.shape[-1]  # Audio features (1 for mono)
    hidden_size = 64
    output_size = y.shape[-1]  # Number of parameters
    
    # Create model
    model = create_mtsf_model(input_size, hidden_size, output_size)
    
    # Loss and optimizer
    loss_fn = nn.MSELoss()  # Regression loss for parameter prediction
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model created: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
    
    # Training loop
    num_epochs = 10
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            forecast, _ = model(batch_x)
            loss = loss_fn(forecast, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Test the model
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch_x, batch_y in test_loader:
            forecast, _ = model(batch_x)
            loss = loss_fn(forecast, batch_y)
            test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "mtsf_model.pt")
    print("Model saved as 'mtsf_model.pt'")

def predict_parameters(model, audio_file_path):
    """Predict parameters for a given audio file"""
    try:
        # Load audio file
        audio_data, sample_rate = sf.read(str(audio_file_path))
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        
        # Pad to match training data length
        if len(audio_tensor) < model.rnn.input_size:
            audio_tensor = torch.cat([audio_tensor, torch.zeros(model.rnn.input_size - len(audio_tensor))])
        elif len(audio_tensor) > model.rnn.input_size:
            audio_tensor = audio_tensor[:model.rnn.input_size]
        
        # Add batch and channel dimensions
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        
        # Predict
        model.eval()
        with torch.no_grad():
            forecast, _ = model(audio_tensor)
            return forecast.squeeze().numpy()
    
    except Exception as e:
        print(f"Error predicting parameters for {audio_file_path}: {e}")
        return None

# Function to predict parameters for new audio files (not run automatically)
def generate_parameters_for_new_file(audio_file_path):
    """Generate parameters for a new audio file using trained model"""
    if 'model' in locals():
        predicted_params = predict_parameters(model, audio_file_path)
        if predicted_params is not None:
            print(f"Predicted parameters for {audio_file_path}:")
            print(predicted_params)
            return predicted_params
    else:
        print("Model not trained yet. Please run training first.")
        return None
