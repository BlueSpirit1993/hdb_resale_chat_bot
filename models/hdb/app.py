from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from main import Model

app = FastAPI()

pipe = Model()
pipe.load_model()
app.state.model = pipe

# Allow all origins (for development only)
origins = ["*"]

# Recommended setup for production:
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
#     "https://your-frontend-domain.com"
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allows all origins
    allow_credentials=True,
    allow_methods=["*"],    # allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],    # allows all headers
)
@app.get("/")
def root():
    return {"Name": "KT"}

@app.get("/predict")
def prediction(block, street_name, town, flat_type, storey_range, floor_area_sqm,
               remaining_lease_year, month_year):
    query = {
    "block": block,
    "street_name": street_name,
    "town": town,
    "flat_type": flat_type,
    "storey_range": storey_range,
    "floor_area_sqm": floor_area_sqm,
    "remaining_lease_year": remaining_lease_year,
    "month_year": month_year
}
    pred_price = app.state.model.get_preds(query)
    return {"HDB Price": pred_price[0]}
