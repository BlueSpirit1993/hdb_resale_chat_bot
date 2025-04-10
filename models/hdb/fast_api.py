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

@app.get("/predict")
def predict(
        block: int,
        street_name: str,
        town: str,
        flat_type: str,
        storey_range: str,
        floor_area_sqm: int,
        remaining_lease_year: int,
        month_year: str
    ):

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

    #pred = model.get_preds(query)

    pred = app.state.model.predict(query)
    return {'price': pred}



@app.get("/")
def root():
    return {'greeting': "Hello"}
