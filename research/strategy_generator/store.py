from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List

import data.database as database
from research.strategy_generator.types import ResearchRunSummary, StrategyEvaluation


class StrategyResearchStore:
    """Persists research runs, candidate evaluations, and active deployments."""

    def start_run(
        self,
        *,
        run_id: str,
        symbol: str,
        timeframe: str,
        candidate_count: int,
        notes: Dict[str, object] | None = None,
    ) -> database.ResearchRun:
        session = database.get_session()
        try:
            row = database.ResearchRun(
                run_id=run_id,
                symbol=symbol,
                timeframe=timeframe,
                mode="PAPER",
                status="running",
                candidate_count=int(candidate_count),
                notes_json=dict(notes or {}),
                started_at=datetime.utcnow(),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row
        finally:
            session.close()

    def save_evaluations(self, run_id: str, evaluations: Iterable[StrategyEvaluation]) -> List[database.ResearchStrategy]:
        session = database.get_session()
        rows: List[database.ResearchStrategy] = []
        try:
            for evaluation in evaluations:
                payload = evaluation.candidate.to_payload()
                row = (
                    session.query(database.ResearchStrategy)
                    .filter(database.ResearchStrategy.strategy_id == evaluation.candidate.strategy_id)
                    .first()
                )
                if row is None:
                    row = database.ResearchStrategy(
                        strategy_id=evaluation.candidate.strategy_id,
                        run_id=run_id,
                        symbol=evaluation.candidate.symbol,
                        timeframe=evaluation.candidate.timeframe,
                        definition_json=payload,
                    )
                    session.add(row)

                row.run_id = run_id
                row.symbol = evaluation.candidate.symbol
                row.timeframe = evaluation.candidate.timeframe
                row.definition_json = payload
                row.indicators_json = {"indicators": evaluation.candidate.indicators}
                row.train_metrics_json = evaluation.train_metrics.to_payload()
                row.test_metrics_json = evaluation.test_metrics.to_payload()
                row.walk_forward_json = {
                    "folds": [fold.to_payload() for fold in evaluation.walk_forward_folds],
                }
                row.score = float(evaluation.test_metrics.score)
                row.rank_percentile = float(evaluation.rank_percentile)
                row.selected = bool(evaluation.selected)
                row.deployed = bool(evaluation.deployed)
                row.status = "accepted" if evaluation.test_metrics.passed else "rejected"
                row.failure_reason = str(evaluation.test_metrics.failure_reason or "")
                row.updated_at = datetime.utcnow()
                rows.append(row)

            session.commit()
            return rows
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def deploy_top_strategies(
        self,
        *,
        run_id: str,
        strategies: Iterable[StrategyEvaluation],
    ) -> List[database.ResearchDeployment]:
        session = database.get_session()
        deployments: List[database.ResearchDeployment] = []
        strategy_list = list(strategies)
        try:
            if strategy_list:
                symbol = strategy_list[0].candidate.symbol
                timeframe = strategy_list[0].candidate.timeframe
                (
                    session.query(database.ResearchDeployment)
                    .filter(database.ResearchDeployment.symbol == symbol)
                    .filter(database.ResearchDeployment.timeframe == timeframe)
                    .filter(database.ResearchDeployment.is_active.is_(True))
                    .update({"is_active": False}, synchronize_session=False)
                )
                session.flush()

            for evaluation in strategy_list:
                deployment = database.ResearchDeployment(
                    run_id=run_id,
                    strategy_id=evaluation.candidate.strategy_id,
                    symbol=evaluation.candidate.symbol,
                    timeframe=evaluation.candidate.timeframe,
                    deployment_mode="PAPER",
                    rank_percentile=float(evaluation.rank_percentile),
                    score=float(evaluation.test_metrics.score),
                    is_active=True,
                    deployed_at=datetime.utcnow(),
                    definition_json=evaluation.candidate.to_payload(),
                )
                session.add(deployment)
                deployments.append(deployment)
            session.commit()
            return deployments
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def complete_run(self, summary: ResearchRunSummary) -> None:
        session = database.get_session()
        try:
            row = (
                session.query(database.ResearchRun)
                .filter(database.ResearchRun.run_id == summary.run_id)
                .first()
            )
            if row is None:
                return
            row.status = "completed"
            row.accepted_count = int(summary.accepted_count)
            row.selected_count = int(summary.selected_count)
            row.deployed_count = int(summary.deployed_count)
            row.completed_at = summary.completed_at or datetime.utcnow()
            row.notes_json = dict(summary.notes)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
