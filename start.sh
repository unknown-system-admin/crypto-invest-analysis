#!/bin/bash
cd /app
python -m uvicorn monitor.main:app --host 0.0.0.0 --port 8000 &
exec streamlit run app.py --server.port 8501 --server.address 0.0.0.0
