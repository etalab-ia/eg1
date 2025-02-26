"""ModeRaw schema for users given output

Revision ID: 7e2ab5bde05b
Revises: 3473db3ff969
Create Date: 2025-02-21 19:51:30.316070

"""

from io import StringIO
from typing import Sequence, Union

from alembic import op
import pandas as pd
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7e2ab5bde05b"
down_revision: Union[str, None] = "3473db3ff969"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("datasets", "has_output")
    op.add_column("models", sa.Column("has_raw_output", sa.Boolean(), nullable=True))
    # ### end Alembic commands ###

    # Add default values
    op.execute(sa.text("UPDATE models SET has_raw_output = FALSE WHERE has_raw_output IS NULL"))
    # Fix "output" datast
    db = op.get_bind()
    result = db.execute(sa.text("SELECT id, df FROM datasets"))
    for row in result:
        df = pd.read_json(StringIO(row.df))
        if not ("output" in df.columns and df["output"].notna().any()):
            continue

        # Fetch experiments related to the current dataset
        experiments = db.execute(
            sa.text("SELECT id FROM experiments WHERE dataset_id = :dataset_id"),
            {"dataset_id": row.id},
        )

        for experiment in experiments:
            db.execute(
                sa.text("""
                        UPDATE models
                        SET has_raw_output = :has_raw_output
                        WHERE id = (SELECT model_id FROM experiments WHERE id = :experiment_id)
                        """),
                {"has_raw_output": True, "experiment_id": experiment["id"]},
            )

            for num_line, row in df.iterrows():
                # Insert into answers table
                db.execute(
                    sa.text("""
                        INSERT INTO answers (created_at, answer, num_line, experiment_id)
                        VALUES (NOW(), :answer, :num_line, :experiment_id)
                    """),
                    {
                        "answer": row["output"],
                        "num_line": num_line,
                        "experiment_id": experiment["id"],
                    },
                )


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("models", "has_raw_output")
    op.add_column("datasets", sa.Column("has_output", sa.BOOLEAN(), autoincrement=False, nullable=True))
    # ### end Alembic commands ###
