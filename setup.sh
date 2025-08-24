#!/bin/bash

echo "🚀 Mise à jour du système..."
sudo apt-get update -y
sudo apt-get install -y build-essential cmake libssl-dev libffi-dev python3.11 python3.11-venv python3.11-dev git wget

echo "🐍 Création de l'environnement virtuel avec Python 3.11..."
python3.11 -m venv venv311
source venv311/bin/activate

echo "📦 Mise à jour de pip..."
pip install --upgrade pip

echo "📥 Installation des dépendances Python..."
pip install --no-cache-dir -r requirements.txt

echo "✅ Tout est prêt !"
echo "Pour activer l'environnement virtuel à l'avenir :"
echo "source venv311/bin/activate"

