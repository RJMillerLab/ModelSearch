# Interactive Demo

Web interface to compare Card2Card (dense semantic) vs Card2Tab2Card (table-based) search.

## Local Development

```bash
# Terminal 1: Start backend
python -m src.demo.backend

# Terminal 2: Start frontend
python -m src.demo.frontend

# Open http://localhost:5001 in browser
```

## Remote Server (SLURM + SSH Tunnel)

### On Remote Server

```bash
# 1. Submit backend
sbatch scripts/run_backend.sh

# 2. Check hostname from output (look for "Hostname: xxx")
tail -f logs/backend_*.out

# 3. Submit frontend (use the hostname from step 2)
BACKEND_HOSTNAME=watgpu308 sbatch scripts/run_frontend.sh
```

### On Local Machine

```bash
# Get hostname from backend output, then create SSH tunnel
# Replace watgpu308 with actual hostname from tail output
ssh -fN -L 5000:watgpu308:5000 -L 5001:watgpu308:5001 your_username@watgpu.cs.uwaterloo.ca
```

**Open in local browser:** `http://localhost:5001`

**Note:** Replace `watgpu308` with the actual hostname shown in `tail -f logs/backend_*.out` output.

## API Endpoints

- `POST /api/search` - Start search pipeline
  - Input: `{"query": "text query", "top_k": 20}`
  - Output: `{"status": "started", "job_id": "uuid"}`

- `GET /api/status/<job_id>` - Get job status and logs
- `GET /api/results/<job_id>` - Get final results
- `GET /api/logs/<job_id>` - Stream logs (SSE)

## Output Format

Results saved to `data/compare_search_{job_id}.json`:
- `card2card_results`: Dense semantic search results
- `card2tab2card_results`: Table-based search results (single_column, keyword, unionable)
- `comparison`: Overlap analysis between methods
