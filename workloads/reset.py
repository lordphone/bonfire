"""Measured reset for an isolated performance simulator, never correctness runs."""


def reset_isolated_task(store, task_id, inventory, *, expected_epoch):
    with store.transaction() as connection:
        task = store.task(connection, task_id)
        if connection.execute("SELECT count(*) FROM tasks").fetchone()[0] != 1:
            raise ValueError("performance reset requires an isolated single-task database")
        if task["owner_epoch"] != expected_epoch:
            raise ValueError("reset epoch is fenced")
        for table in ("tool_effects", "operation_identity", "operations", "log_records", "inventory"):
            connection.execute(f"DELETE FROM {table} WHERE task_id=?", (task_id,))
        connection.executemany("INSERT INTO inventory VALUES (?,?,?)", [(task_id, sku, count) for sku, count in inventory.items()])
        connection.execute("UPDATE tasks SET halted=0,context_record_id=NULL WHERE task_id=?", (task_id,))
