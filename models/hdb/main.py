import numpy as np
import pandas as pd
import pickle
import requests
from sklearn.neighbors import BallTree

class Model:
    def __init__(self):
        self.categorical_cols = [
            "block", #user to give
            "street_name", #user to give
            "town", #user to give
            "flat_type", #user to give
            "storey_range", #user to give - Low, Mid, High
            "region", #get_region func below
            "line", #get_nearest_mrt_info func below
            "nearest_mrt_name", #get_nearest_mrt_info func below
            "nearest_sch_name" # get_nearest_school_info below
        ]
        self.numeric_cols = [
            "floor_area_sqm", #user to give, but if not given, refer to get_sqm_func below
            "remaining_lease_year", #user to give
            "distance_to_nearest_sch", # get_nearest_school_info
            "distance_to_nearest_mrt" #get_nearest_mrt_info func below
        ]
        self.date_cols = ["month_sin", "month_cos", "salesYear_fr_2017"] #user to give month, use convert_month_features func below to convert

        self.pipeline = None

        self.gdsch=pd.read_csv("gdsch.csv")

        self.mrt=pd.read_csv("mrt.csv")

        self.region = {
            "SENGKANG": "North-East",
            "PUNGGOL": "North-East",
            "HOUGANG": "North-East",
            "WOODLANDS": "North",
            "YISHUN": "North",
            "SEMBAWANG": "North",
            "TAMPINES": "East",
            "PASIR RIS": "East",
            "BEDOK": "East",
            "GEYLANG": "East",
            "MARINE PARADE": "East",
            "JURONG WEST": "West",
            "JURONG EAST": "West",
            "CHOA CHU KANG": "West",
            "BUKIT BATOK": "West",
            "BUKIT PANJANG": "West",
            "CLEMENTI": "West",
            "ANG MO KIO": "Central",
            "BISHAN": "Central",
            "TOA PAYOH": "Central",
            "KALLANG/WHAMPOA": "Central",
            "QUEENSTOWN": "Central",
            "BUKIT MERAH": "Central",
            "SERANGOON": "Central",
            "CENTRAL AREA": "Central",
            "BUKIT TIMAH": "Central"
            }

        self.sqm = {
            "1 ROOM":31.0,
            "2 ROOM":47.0,
            "3 ROOM":67.0,
            "4 ROOM":93.0,
            "5 ROOM": 110.0,
            "EXECUTIVE": 146.0,
            "MULTI-GENERATION":164.0
        }

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
        if not result:
            raise ValueError(f"No coordinates found for address: {add}")
        return float(result["LATITUDE"]), float(result["LONGITUDE"])

    def get_nearest_school_info(self, block, street_name):

        lat, lon = self.get_coordinates(block, street_name)
        gd_sch_df = self.gdsch

        addr_coords = np.radians(np.array([[lat, lon]]))
        sch_coords = np.radians(gd_sch_df[["latitude", "longitude"]].values)

        tree = BallTree(sch_coords, metric="haversine")
        distances, indices = tree.query(addr_coords, k=1)

        earth_radius = 6371000
        distance_s = distances[0][0] * earth_radius
        idx = indices[0][0]
        school_name = gd_sch_df.iloc[idx]["school_name"]

        return {
            "walk_distance_to_school": distance_s,
            "nearest_sch_name": school_name
            }


    def get_nearest_mrt_info(self, block, street_name):
        lat, lon = self.get_coordinates(block, street_name)
        mrt_df = self.mrt

        addr_coords = np.radians(np.array([[lat, lon]]))
        mrt_coords = np.radians(mrt_df[["latitude", "longitude"]].values)

        tree = BallTree(mrt_coords, metric="haversine")
        distances, indices = tree.query(addr_coords, k=1)

        earth_radius = 6371000
        distance_m = distances[0][0] * earth_radius
        idx = indices[0][0]
        mrt_name=mrt_df.iloc[idx]["mrt"]
        mrt_line = mrt_df.iloc[idx]["line"]

        return {
            "walk_distance_to_mrt": distance_m,
            "nearest_mrt_name": mrt_name,
            "line": mrt_line
            }

    def get_region(self, region):
        return self.region.get(region, None)

    def get_sqm(self, sqm):
        return self.sqm.get(sqm, None)

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

        preprocessed_user_query = {}

        for col in self.categorical_cols:
            if col in user_query:
                preprocessed_user_query[col] = user_query[col]

        for col in self.numeric_cols:
            if col in user_query:
                preprocessed_user_query[col] = user_query[col]

        date_features = self.convert_month_features(user_query["month_year"])
        preprocessed_user_query["month_sin"] = date_features["month_sin"]
        preprocessed_user_query["month_cos"] = date_features["month_cos"]
        preprocessed_user_query["salesYear_fr_2017"] = date_features["salesYear_fr_2017"]

        block = user_query["block"]
        street_name = user_query["street_name"]

        mrt_info = self.get_nearest_mrt_info(block, street_name)
        school_info = self.get_nearest_school_info(block, street_name)
        preprocessed_user_query["line"] = mrt_info["line"]
        preprocessed_user_query["nearest_mrt_name"] = mrt_info["nearest_mrt_name"]
        preprocessed_user_query["nearest_sch_name"] = school_info["nearest_sch_name"]
        preprocessed_user_query["walk_distance_to_mrt"] = mrt_info["walk_distance_to_mrt"]
        preprocessed_user_query["walk_distance_to_school"] = school_info["walk_distance_to_school"]
        preprocessed_user_query["GDP"] = 1000

        town = user_query.get("town", "").upper()
        preprocessed_user_query["region"] = self.get_region(town)

        if "floor_area_sqm" not in user_query or pd.isna(user_query["floor_area_sqm"]):
            flat_type = user_query.get("flat_type", "").upper()
            preprocessed_user_query["floor_area_sqm"] = self.get_sqm(flat_type)

        print(preprocessed_user_query)

        return pd.Series(preprocessed_user_query)

    def load_model(self):
        with open("best_lgbm_pipeline.pkl", "rb") as f:
            self.pipeline = pickle.load(f)

    def get_preds(self, user_query):
        x = self.preprocess(user_query)
        return self.pipeline.predict(x.to_frame().T)

    def fit(self, X, y):
        self.pipeline.fit(X,y)
