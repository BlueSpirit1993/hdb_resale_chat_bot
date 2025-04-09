import numpy as np
import pandas as pd
import pickle
from typing import Dict
import requests
from sklearn.neighbors import BallTree

class Model:
    def __init__(self):
        self.categorical_cols = [
            "block",
            "street_name",
            "flat_type",
            "flat_model",
            "region",
            "storey_range",
            "line"
        ]
        self.numeric_cols = [
            "floor_area_sqm",
            "remaining_lease_year",
            "walk_distance_to_school",
            "walk_distance_to_mrt",
        ]
        self.date_cols = ["month_sin", "month_cos", "salesYear_fr_2017"]

        self.pipeline = None

        self.schools=pd.read_csv("")

        self.mrt=


    def get_coordinates(self, block, street_name):
        add = block + " " + street_name
        url = "https://www.onemap.gov.sg/api/common/elastic/search"
        params = {
            "searchVal": add,
            "returnGeom": "Y",
            "getAddrDetails": "N",
            "pageNum": 1
        }
        response = requests.get(url, params=params)
        data = response.json()
        result = data["results"][0]
        return float(result["LATITUDE"]), float(result["LONGITUDE"])

    def get_nearest_school_info(self, address):

        lat, lon = self.get_coordinates(address)
        gd_sch_df = pd.DataFrame(self.gd_schools)

        addr_coords = np.radians(np.array([[lat, lon]]))
        sch_coords = np.radians(gd_sch_df[["latitude", "longitude"]].values)

        tree = BallTree(sch_coords, metric="haversine")
        distances, indices = tree.query(addr_coords, k=1)

        earth_radius = 6371000
        distance_s = distances[0][0] * earth_radius
        idx = indices[0][0]
        school_name = gd_sch_df.iloc[idx]["school_name"]

        return {
            "distance": distance_s,
            "school_name": school_name
            }

    def get_nearest_mrt_info(self, address):

    lat, lon = self.get_coordinates(address)
    mrt_df = pd.DataFrame(self.mrt)

    addr_coords = np.radians(np.array([[lat, lon]]))
    mrt_coords = np.radians(mrt_df[["Latitude", "Longitude"]].values)

    tree = BallTree(mrt_coords, metric="haversine")
    distances, indices = tree.query(addr_coords, k=1)

    earth_radius = 6371000
    distance_m = distances[0][0] * earth_radius
    idx = indices[0][0]
    mrt_name=mrt_df.iloc[idx]["STN_NAME"]
    mrt_line = mrt_df.iloc[idx]["line"]

    return {
        "distance": distance_m,
        "mrt_name": mrt_name,
        "line": line
        }

    def get_mrt_line_tag(self, mrt_name):
        return self.mrt_name_to_line_tag.get(mrt_name, None)

    def convert_month_features(self, month_year): #month_year should be in MM-YYYY format
        month, year = map(int, month_year.split("-"))
        month_sin = np.sin((month - 1) * (2 * np.pi / 12))
        month_cos = np.cos((month - 1) * (2 * np.pi / 12))
        sales_year_fr_2017 = year - 2017
        return {
            "month_sin": month_sin,
            "month_cos": month_cos,
            "salesYear_fr_2017": sales_year_fr_2017
        }

    def preprocess(self, user_query: dict):
        # TODO: Call one map to get mrt station name
        # TODO: Get walking dist, mrt station line, primary schools
        #       mrt_station_line = self.get_mrt_line_tag(mrt_station_name)
        # TODO:
        # preprocessed_user_query = {
        #    "line": mrt_station_line
        # }
        return preprocessed_user_query

    def load_model(self):
        with open("lightgbm_pipeline.pkl", "rb") as f:
            self.pipeline = pickle.load(f)



    def _encode_date(self, df: pd.DataFrame) -> pd.DataFrame:
        df["month_sin"] = np.sin((df["month_num"] - 1) * (2 * np.pi / 12))
        df["month_cos"] = np.cos((df["month_num"] - 1) * (2 * np.pi / 12))
        df["salesYear_fr_2017"] = df["year"] - 2017
        return df

    def _get_geodata(self, address: str) -> Dict:
        # TODO: Implement OneMap API call to get lat/lon
        # This is a stub
        return {
            "lat": 1.3521,
            "lon": 103.8198,
            "nearest_mrt": "Ang Mo Kio",
            "nearest_school": "Mayflower Primary",
            "walk_distance_to_mrt": 600,
            "walk_distance_to_school": 300,
        }

    def preprocess(self, user_query: Dict) -> pd.DataFrame:
        # Extract date features
        user_query["month_num"] = int(user_query["month"])
        user_query["year"] = int(user_query["year"])
        df = pd.DataFrame([user_query])
        df = self._encode_date(df)

        # Get geodata
        geo = self._get_geodata(user_query["add"])
        df["walk_distance_to_mrt"] = geo["walk_distance_to_mrt"]
        df["walk_distance_to_school"] = geo["walk_distance_to_school"]
        df["line"] = self.get_mrt_line_tag(geo["nearest_mrt"])

        return df[self.categorical_cols + self.numeric_cols + self.date_cols]

    def get_preds(self, user_query: Dict):
        x = self.preprocess(user_query)
        preds = self.pipeline.predict(x)
        return preds[0]  # Assuming single prediction


def get_predictions(town: str, storey: str, add: str, month: int, year: int, floor_area_sqm: float, lease_year: float):
    assert storey in ["low", "middle", "high"], "Please choose between low, middle, or high storey."

    query = {
        "town": town,
        "storey_range": storey,
        "add": add,
        "month": month,
        "year": year,
        "flat_type": "4 ROOM",  # Example static inputs
        "flat_model": "Improved",
        "region": "North",
        "floor_area_sqm": floor_area_sqm,
        "remaining_lease_year": lease_year
    }

    return model.get_preds(query)
