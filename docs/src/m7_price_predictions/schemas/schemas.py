import pandera as pa
from pandera.engines import pandas_engine as pe

FormattedTradesDF = pa.DataFrameSchema(
    columns={
        "date_on": pa.Column(pe.DateTime(tz="utc"), nullable=False),
        "date_start": pa.Column(pe.DateTime(tz="utc"), nullable=False),
        "ets_hour_price": pa.Column(float, nullable=True),
        "m7_public_trades_2h": pa.Column(float, nullable=True),
        "m7_public_trades_4h": pa.Column(float, nullable=True),
        "m7_public_trades_hh": pa.Column(float, nullable=True),
        "4h_start": pa.Column(pe.DateTime(tz="Europe/London"), nullable=False),
        "2h_start": pa.Column(pe.DateTime(tz="Europe/London"), nullable=False),
    },
    add_missing_columns=True,
    coerce=True,
)

ShapedTradesDF = pa.DataFrameSchema(
    columns={
        "date_on": pa.Column(pe.DateTime(tz="utc"), nullable=False),
        "date_start": pa.Column(pe.DateTime(tz="utc"), nullable=False),
        "ets_hour_price": pa.Column(float, nullable=True),
        "traded_price": pa.Column(float, nullable=True),
    },
    add_missing_columns=True,
    coerce=True,
)
