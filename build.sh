python3.14 -m venv venv_project && source venv_project/bin/activate
pip install -r requirements.txt

python3.14  db.py add-user adminuser      # asks for a password
python3.14 app.py