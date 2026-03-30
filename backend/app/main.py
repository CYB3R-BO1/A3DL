from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import agent, attack, defend, detect, experiments, health, report, train
from app.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.app_version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(attack.router, prefix="/api", tags=["attack"])
    app.include_router(detect.router, prefix="/api", tags=["detect"])
    app.include_router(defend.router, prefix="/api", tags=["defend"])
    app.include_router(report.router, prefix="/api", tags=["report"])
    app.include_router(agent.router, prefix="/api", tags=["agent"])
    app.include_router(train.router, prefix="/api", tags=["train"])
    app.include_router(experiments.router, prefix="/api", tags=["experiments"])
    return app


app = create_app()
