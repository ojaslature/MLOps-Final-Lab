import pickle
import os

def check_ml_artifacts():
    paths = ["models/model.pkl", "models/scaler.pkl"]
    
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                print(f"✅ {path} loaded successfully. Type: {type(obj)}")
            except Exception as e:
                print(f"❌ {path} failed to load: {e}")
        else:
            print(f"⚠️ {path} is missing!")

if __name__ == "__main__":
    check_ml_artifacts()