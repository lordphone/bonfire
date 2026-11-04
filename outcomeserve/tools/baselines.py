"""Explicit unsafe retry identities for comparison; never used by OutcomeServe."""

from outcomeserve.tools import ToolService


class AttemptToolService:
    def __init__(self, store, *, after_commit=None):
        self.store = store
        self.service = ToolService(store, after_commit=after_commit)

    def submit(self, *, logical_step_id, attempt, **request):
        original_op = request["op_id"]
        request = {**request, "step_id": f"{logical_step_id}:attempt-{attempt}",
                   "op_id": f"{original_op}:attempt-{attempt}"}
        with self.store.transaction() as connection:
            existing = connection.execute("SELECT logical_step_id,execution_attempt FROM operation_identity WHERE task_id=? AND op_id=?",
                                          (request["task_id"], request["op_id"])).fetchone()
            if existing is not None and tuple(existing) != (logical_step_id, str(attempt)):
                raise ValueError("baseline identity collision")
            connection.execute("INSERT OR IGNORE INTO operation_identity VALUES (?,?,?,?)",
                               (request["task_id"], request["op_id"], logical_step_id, str(attempt)))
        return self.service.submit(**request)
