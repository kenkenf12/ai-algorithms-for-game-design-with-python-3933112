if [ -f requirements.txt ]; then
  pip install --user -r requirements.txt
  code --install-extension extensions/cat-trap-game-ui-0.0.3.vsix
fi
