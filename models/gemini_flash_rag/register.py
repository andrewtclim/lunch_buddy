# models/gemini_flash_rag/register.py
# Registers the gemini_flash_rag_v1 pipeline with MLflow.
#
# What gets registered:
#   - Parameters: all knobs that define this version of the pipeline
#   - A pyfunc model wrapper: a callable Python class MLflow can serialize and load
#   - A sample input/output JSON artifact for documentation and schema reference
#
# Run this script once to register. After registration, FastAPI can load the model via:
#   mlflow.pyfunc.load_model("models:/gemini_flash_rag_v1/1")
#
# The pyfunc wrapper imports recommend.py and user_prefs.py at predict() time,
# so those files must be present wherever the model is loaded.

import os
import json
import mlflow
import mlflow.pyfunc
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# load env vars from models/.env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://35.232.122.64:5000")
EXPERIMENT_NAME     = "lunch-buddy-rag"
MODEL_NAME          = "gemini_flash_rag_v1"   # name in the MLflow Model Registry

# pipeline parameters -- these are what distinguish v1 from future versions
PARAMS = {
    "embed_model":     "text-embedding-004",  # Vertex AI embedding model
    "gen_model":       "gemini-2.5-flash",    # Gemini model for recommendations
    "alpha":           0.3,    # EMA weight -- how strongly each pick shifts pref_vec
    "beta":            0.3,    # mood blend weight -- how strongly mood shifts query_vec
    "similarity_metric": "cosine",            # distance function used in pgvector
    "top_k_retrieval": 40,     # dishes fetched from Supabase before filtering
    "top_k_gemini":    5,      # candidates passed to Gemini after filtering
}


# ---------------------------------------------------------------------------
# pyfunc wrapper
# ---------------------------------------------------------------------------

class GeminiFlashRAG(mlflow.pyfunc.PythonModel):
    # MLflow pyfunc wrapper around our recommendation pipeline.
    # MLflow calls predict() when the model is loaded and used for inference.
    #
    # Input dict keys:
    #   pref_vec           -- list of 768 floats (user's preference vector)
    #   preference_summary -- one-sentence taste description (string)
    #   allergens          -- list of allergen strings, e.g. ["gluten"]
    #   date_str           -- "YYYY-MM-DD" string, which day to pull dishes for
    #   daily_mood         -- optional string, e.g. "something light today"
    #   table              -- optional, "daily_menu" or "backfill_menu"
    #
    # Output dict keys:
    #   recommendations -- list of dicts, each with dish_name, dining_hall, reason
    #   alternatives    -- list of 2 diverse options beyond the top 3

    def predict(self, context, model_input: dict) -> dict:
        # import here (not at module top) so MLflow can serialize this class
        # without needing all dependencies installed at registration time
        from recommend import recommend

        # unpack required fields from the input dict
        pref_vec           = np.array(model_input["pref_vec"], dtype=float)
        preference_summary = model_input["preference_summary"]
        allergens          = model_input.get("allergens", [])
        date_str           = model_input["date_str"]

        # unpack optional fields -- both default to safe values if not provided
        daily_mood = model_input.get("daily_mood", None)
        table      = model_input.get("table", "daily_menu")

        # recommend() now returns (recs, alts, query_vec)
        recs, alts, _ = recommend(
            pref_vec=pref_vec,
            preference_summary=preference_summary,
            user_allergens=allergens,
            date_str=date_str,
            daily_mood=daily_mood,
            table=table,
        )

        return {"recommendations": recs, "alternatives": alts}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def build_sample_io() -> dict:
    # a concrete example of what goes into predict() and what comes back out
    # saved as an artifact so anyone reading the MLflow run knows the exact schema
    return {
        "input": {
            "pref_vec":           [0.0] * 768,   # placeholder -- 768 zeros, real vec is private
            "preference_summary": "loves spicy Korean food and chicken",
            "allergens":          ["shellfish"],
            "date_str":           "2026-04-11",
            "daily_mood":         "something light today",
            "table":              "daily_menu",
        },
        "output": {
            "recommendations": [
                {
                    "dish_name":   "Gochujang Pork",
                    "dining_hall": "Lakeside",
                    "reason":      "Matches your Korean spice preference; lighter than a full rice bowl.",
                },
                {
                    "dish_name":   "Grilled Chicken Salad",
                    "dining_hall": "Wilbur",
                    "reason":      "High protein chicken in a light format -- fits today's mood.",
                },
                {
                    "dish_name":   "Kimchi Fried Rice",
                    "dining_hall": "Arrillaga",
                    "reason":      "Bold Korean flavors; portion is moderate.",
                },
            ],
            "alternatives": [
                {
                    "dish_name":   "Butter Chicken",
                    "dining_hall": "Wilbur",
                    "reason":      "Indian spiced, milder -- different cuisine worth exploring.",
                },
                {
                    "dish_name":   "Grilled Salmon",
                    "dining_hall": "Florence Moore",
                    "reason":      "Lighter protein, completely different style from usual picks.",
                },
            ],
        },
    }



def register():
    # point MLflow at our remote tracking server
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)   # creates experiment if it doesn't exist

    with mlflow.start_run(run_name=f"{MODEL_NAME}_registration"):

        # 1 -- log all pipeline parameters so this run is fully reproducible
        mlflow.log_params(PARAMS)

        # 2 -- save the sample input/output both locally and to MLflow/GCS
        # write locally first (mlflow.log_dict() fails on Mac with mlflow 3.x
        # because it tries to create /home/... temp paths that don't exist)
        sample_path = Path(__file__).parent / "sample_io.json"
        with open(sample_path, "w") as f:
            json.dump(build_sample_io(), f, indent=2)
        print(f"Sample I/O saved locally to {sample_path}", flush=True)

        # then upload to MLflow -- log_artifact() takes an existing file so it
        # avoids the temp-path bug; wrapped in try/except as a safety net
        try:
            mlflow.log_artifact(str(sample_path))
            print("Sample I/O uploaded to MLflow artifact store (GCS).", flush=True)
        except Exception as e:
            print(f"  [artifact upload skipped -- local copy preserved: {e}]", flush=True)

        # 3 -- save the pyfunc model locally first, then upload as a plain artifact
        # mlflow.pyfunc.log_model() uses a new 3.x endpoint not supported by the 2.x
        # server, so we use the older save -> log_artifacts -> register_model pattern
        local_model_dir = Path(__file__).parent / "_model_artifact"
        mlflow.pyfunc.save_model(
            path=str(local_model_dir),   # save to a local temp folder
            python_model=GeminiFlashRAG(),   # our pyfunc wrapper
            # note: code_path removed -- mlflow 3.x dropped this param from save_model
            # recommend.py and user_prefs.py must be importable from wherever the
            # model is loaded (they live in models/gemini_flash_rag/ in the repo)
        )

        # upload the local model folder to the run's artifact store (GCS via server)
        # artifact_path="model" means it lands at gs://mlflow-model-artifacts/.../model/
        mlflow.log_artifacts(str(local_model_dir), artifact_path="model")

        # register the uploaded model in the MLflow Model Registry
        # runs:/ URI points MLflow to the artifact we just uploaded
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, MODEL_NAME)

        print(f"Registered '{MODEL_NAME}' -- run ID: {run_id}", flush=True)
        print(f"MLflow UI: {MLFLOW_TRACKING_URI}/#/experiments", flush=True)
        print(f"Load with: mlflow.pyfunc.load_model('models:/{MODEL_NAME}/latest')", flush=True)

        # clean up the local model folder -- it's now safely uploaded to GCS
        import shutil
        shutil.rmtree(local_model_dir)


if __name__ == "__main__":
    register()
