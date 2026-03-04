# Scheduled Paper Ingestion

Automated periodic paper fetching from arXiv using cron-style scheduling.

## Status

🔲 **Not Implemented** – Skeleton available

## Overview

Scheduled ingestion enables automatic discovery of new papers on a regular basis, keeping your research corpus up-to-date without manual intervention.

## Files

- [scheduler/jobs.py](../src/papertrail/scheduler/jobs.py) – Scheduler implementation

## Features to Implement

### Job Configuration

```python
from papertrail.scheduler.jobs import IngestionJob, PaperScheduler

# Create a daily AI papers job
job = IngestionJob(
    job_id="daily_ai",
    categories=["cs.AI", "cs.LG", "cs.CL"],
    max_results=50,
    cron_expression="0 6 * * *",  # 6 AM daily
)

# Add to scheduler
scheduler = PaperScheduler()
scheduler.add_job(job)
scheduler.start()
```

### Cron Expressions

| Expression | Description |
|---|---|
| `0 6 * * *` | Every day at 6:00 AM |
| `0 8 * * 1` | Every Monday at 8:00 AM |
| `0 */4 * * *` | Every 4 hours |
| `0 6 * * 1-5` | Weekdays at 6:00 AM |
| `0 0 1 * *` | First day of each month |

### Default Jobs

```python
DEFAULT_JOBS = [
    IngestionJob(
        job_id="daily_ai_papers",
        categories=["cs.AI", "cs.LG", "cs.CL"],
        max_results=50,
        cron_expression="0 6 * * *",
    ),
    IngestionJob(
        job_id="weekly_cv_papers",
        categories=["cs.CV"],
        max_results=100,
        cron_expression="0 8 * * 1",
    ),
]
```

## Implementation Tasks

### Core Scheduler

- [ ] APScheduler integration
- [ ] Cron trigger parsing
- [ ] Job persistence (SQLite/JSON)
- [ ] Graceful shutdown handling
- [ ] Job state recovery on restart

### Job Management

- [ ] Add/remove/update jobs
- [ ] Enable/disable jobs
- [ ] Manual job triggering
- [ ] Job history tracking
- [ ] Error retry logic

### Notifications

- [ ] Email notifications on completion/failure
- [ ] Slack/Discord webhook integration
- [ ] Desktop notifications

### Monitoring

- [ ] Job execution logs
- [ ] Success/failure metrics
- [ ] Last run timestamps
- [ ] Next run predictions

### CLI Integration

- [ ] `papertrail schedule list` – List all jobs
- [ ] `papertrail schedule add` – Add new job
- [ ] `papertrail schedule remove` – Remove job
- [ ] `papertrail schedule run` – Manual trigger
- [ ] `papertrail schedule start` – Start daemon

## CLI Commands

```bash
# List scheduled jobs
papertrail schedule list

# Add a new job
papertrail schedule add \
  --id "nightly_ml" \
  --categories "cs.LG,stat.ML" \
  --max-results 30 \
  --cron "0 2 * * *"

# Remove a job
papertrail schedule remove --id "nightly_ml"

# Run a job immediately
papertrail schedule run --id "daily_ai_papers"

# Start scheduler daemon
papertrail schedule start --foreground

# Check job status
papertrail schedule status
```

## Daemon Mode

### Systemd Service

```ini
# /etc/systemd/system/papertrail-scheduler.service
[Unit]
Description=PaperTrail Paper Ingestion Scheduler
After=network.target

[Service]
Type=simple
User=papertrail
WorkingDirectory=/opt/papertrail
ExecStart=/opt/papertrail/.venv/bin/python -m papertrail.scheduler.jobs
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["python", "-m", "papertrail.scheduler.jobs"]
```

### Supervisor

```ini
[program:papertrail-scheduler]
command=/opt/papertrail/.venv/bin/python -m papertrail.scheduler.jobs
directory=/opt/papertrail
user=papertrail
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/papertrail/scheduler.log
```

## State Persistence

```json
// data/scheduler_state.json
{
  "jobs": {
    "daily_ai_papers": {
      "job_id": "daily_ai_papers",
      "categories": ["cs.AI", "cs.LG", "cs.CL"],
      "max_results": 50,
      "cron_expression": "0 6 * * *",
      "enabled": true,
      "last_run": "2024-03-15T06:00:00Z",
      "run_count": 45,
      "error_count": 1
    }
  }
}
```

## Dependencies

```toml
apscheduler = ">=3.10.0"

# Optional for notifications
slack-sdk = ">=3.27.0"      # Slack notifications
sendgrid = ">=6.11.0"       # Email via SendGrid
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              PaperScheduler                  │
│  ┌─────────────────────────────────────┐    │
│  │         APScheduler Backend          │    │
│  │  ┌─────────┐  ┌─────────┐           │    │
│  │  │  Job 1  │  │  Job 2  │  ...      │    │
│  │  │ (cron)  │  │ (cron)  │           │    │
│  │  └────┬────┘  └────┬────┘           │    │
│  └───────┼────────────┼────────────────┘    │
│          │            │                      │
│          ▼            ▼                      │
│  ┌─────────────────────────────────────┐    │
│  │           Job Executor               │    │
│  │                                      │    │
│  │  ArxivClient → Download → Process   │    │
│  │       → Embed → Index → Notify      │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

## References

- [APScheduler Documentation](https://apscheduler.readthedocs.io/)
- [Cron Expression Generator](https://crontab.guru/)
- [Systemd Service Files](https://www.freedesktop.org/software/systemd/man/systemd.service.html)
