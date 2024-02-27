import pandas as pd
import sqlalchemy

from EDMS.model_optimization import model
from edfp import generate_edfp
from qsar import qsar_model

engine = sqlalchemy.create_engine('qsar_ready_table.db')
query = "qsar_ready_table"
qsar_data = pd.read_sql(query, engine)

qsar_data['generate_edfp'] = qsar_data.apply(lambda row: generate_edfp(model, row), axis=1)

predictions = qsar_model.predict(qsar_data['generate_edfp'])

matched_data = qsar_data[predictions == 'ahr_active']