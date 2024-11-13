# Initialize Git repository
git init

# Add the file to the staging area
git add KMeans.py

# Commit the file with a message
git commit -m "Initial commit of KMeans.py"

# Create a new repository on GitHub and follow these steps:
# 1. Go to https://github.com/new
# 2. Enter your repository name (e.g., `stock-clustering-app`)
# 3. Choose the visibility (public or private)
# 4. Click "Create repository"
# 5. Follow the instructions to push your local repository to GitHub

git remote add origin https://github.com/your-username/stock-clustering-app.git
git branch -M main
git push -u origin main
