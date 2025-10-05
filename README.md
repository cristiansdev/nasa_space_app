# 1️⃣ Clone the repository
git clone https://github.com/<yourusername>/SB_PUBLICATIONS.git
cd SB_PUBLICATIONS

# 2️⃣ Create and activate virtual environment
python3 -m venv env
source env/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Add your environment variables
echo "GOOGLE_API_KEY=YOUR_KEY" >> .env
echo "AWS_ACCESS_KEY_ID=YOUR_AWS_KEY" >> .env
echo "AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET" >> .env

# 5️⃣ Run the app
streamlit run app/Home.py

## NOTA

La documentación sigue en proceso