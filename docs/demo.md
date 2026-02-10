# Interactive Demo

Web interface to compare Card2Card (dense semantic) vs Card2Tab2Card (table-based) search.


```bash
# Terminal 1: Start backend
python -m src.demo.backend

# Terminal 2: Start frontend
python -m src.demo.frontend

# Remote Server (SLURM + SSH Tunnel)
ssh -L 5001:127.0.0.1:5001 -L 5002:127.0.0.1:5002 chippie.cs.uwaterloo.ca
```

**Open in local browser:** `http://localhost:5001`
