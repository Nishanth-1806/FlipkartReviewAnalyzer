#!/bin/bash

# Update system
sudo yum update -y || sudo apt-get update -y

# Install Python and Git
sudo yum install python3 git -y || sudo apt-get install python3 git -y

# Verify Python installation
python3 --version

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r app/requirements.txt

# Download NLTK data (if not already handled in code, but good to ensure)
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create a run script
echo "#!/bin/bash
source venv/bin/activate
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
" > run_app.sh

chmod +x run_app.sh

echo "Setup complete! Run './run_app.sh' to start the app."
