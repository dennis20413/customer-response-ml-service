from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NUMERIC = ["age", "pdays", "previous", "emp_var_rate", "cons_price_idx",
           "cons_conf_idx", "euribor3m", "nr_employed"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "poutcome"]
FEATURES = NUMERIC + CATEGORICAL
EXCLUDED = ["duration", "campaign", "contact", "month", "day_of_week"]
RENAME = {"emp.var.rate": "emp_var_rate", "cons.price.idx": "cons_price_idx",
          "cons.conf.idx": "cons_conf_idx", "nr.employed": "nr_employed"}

Category = Annotated[str, Field(min_length=1, max_length=48, pattern=r"^\S.*$")]


class Customer(BaseModel):
    """Features available before the planned call; additional inputs are rejected."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    age: Annotated[int, Field(ge=16, le=120)]
    pdays: Annotated[int, Field(ge=0, le=999)]
    previous: Annotated[int, Field(ge=0, le=100)]
    emp_var_rate: Annotated[float, Field(ge=-20, le=20)]
    cons_price_idx: Annotated[float, Field(gt=0, le=200)]
    cons_conf_idx: Annotated[float, Field(ge=-100, le=100)]
    euribor3m: Annotated[float, Field(ge=-10, le=30)]
    nr_employed: Annotated[float, Field(gt=0, le=100000)]
    job: Category
    marital: Category
    education: Category
    default: Category
    housing: Category
    loan: Category
    poutcome: Category


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: Annotated[list[Customer], Field(min_length=1, max_length=256)]
