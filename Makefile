dev:
	@echo "http://localhost:8888/lab?token=777"
	@jupyter lab --NotebookApp.token='777' 2>&1 | grep "TODO: 後でリファクタリングする。何も表示したくないため、grepに適当な文字列を渡している。"

activate:
	@echo "To activate the virtual environment, run:"
	@echo "source venv/bin/activate"
