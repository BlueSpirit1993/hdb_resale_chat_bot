from fastapi import FastAPI
import pickle

model_path = "best_lgbm_pipeline.pkl"
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)


app = FastAPI()
app.state.model = loaded_model

@app.get("/predict")
def predict(
        block,
        street_name,
        town,
        flat_type,
        storey_range,
        floor_area_sqm,
        remaining_lease_year,
        month_year
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
