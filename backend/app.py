# =====================================================
# IMPORTS
# =====================================================
import os
import json
import traceback
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from preprocessing import preprocess_signature
from loader import get_embedding


# =====================================================
# CONFIG
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
FRONTEND_FOLDER = os.path.join(BASE_DIR, "..", "frontend")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# =====================================================
# FLASK APP
# =====================================================
app = Flask(
    __name__,
    static_folder=FRONTEND_FOLDER,
    static_url_path=""
)

CORS(app)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# =====================================================
# HELPERS
# =====================================================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_customer_data_path(customer_id):
    return os.path.join(DATA_FOLDER, f"{customer_id}.json")


def save_customer_data(customer_id, customer_name, embeddings):

    embeddings = np.array(embeddings)

    # ----------------------------------
    # Centroid
    # ----------------------------------
    centroid = np.mean(embeddings, axis=0)

    # ----------------------------------
    # Distance of each real signature
    # ----------------------------------
    distances = [
        np.linalg.norm(e - centroid)
        for e in embeddings
    ]

    mean_distance = float(np.mean(distances))
    std_distance = float(np.std(distances))

    data = {
        "customer_id": customer_id,
        "customer_name": customer_name,
        "num_signatures": len(embeddings),
        "centroid": centroid.tolist(),
        "mean_distance": mean_distance,
        "std_distance": std_distance,
        "embeddings": embeddings.tolist()
    }

    with open(get_customer_data_path(customer_id), "w") as f:
        json.dump(data, f, indent=4)


def load_customer_data(customer_id):

    path = get_customer_data_path(customer_id)

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    data["centroid"] = np.array(data["centroid"])
    data["embeddings"] = [np.array(e) for e in data["embeddings"]]

    return data


# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def home():
    return send_from_directory(FRONTEND_FOLDER, "index.html")


@app.route("/assets/<path:path>")
def serve_assets(path):
    return send_from_directory(os.path.join(FRONTEND_FOLDER, "assets"), path)


# =====================================================
# ADD CUSTOMER
# =====================================================
@app.route("/add-customer", methods=["POST"])
def add_customer():

    try:

        customer_name = request.form.get("customerName", "").strip()
        customer_id = request.form.get("customerId", "").strip()

        if not customer_name or not customer_id:
            return jsonify({"success": False, "error": "Customer name and ID required"}), 400

        if os.path.exists(get_customer_data_path(customer_id)):
            return jsonify({"success": False, "error": "Customer already exists"}), 400

        files = request.files.getlist("signatures")

        if len(files) < 6 or len(files) > 12:
            return jsonify({"success": False, "error": "Upload between 6 and 12 signatures"}), 400

        customer_folder = os.path.join(UPLOAD_FOLDER, customer_id)
        os.makedirs(customer_folder, exist_ok=True)

        embeddings = []

        for i, file in enumerate(files):

            if not allowed_file(file.filename):
                return jsonify({"success": False, "error": f"Invalid file type: {file.filename}"}), 400

            ext = file.filename.rsplit(".", 1)[1].lower()

            save_path = os.path.join(customer_folder, f"sig_{i+1}.{ext}")

            file_bytes = file.read()

            with open(save_path, "wb") as f:
                f.write(file_bytes)

            normalized_img, _ = preprocess_signature(
                file_bytes,
                from_bytes=True
            )

            emb = get_embedding(normalized_img)

            embeddings.append(emb)

        save_customer_data(customer_id, customer_name, embeddings)

        return jsonify({
            "success": True,
            "message": f"{customer_name} added successfully",
            "customerId": customer_id,
            "signaturesProcessed": len(embeddings)
        })

    except Exception as e:

        print("\n ERROR in /add-customer")
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =====================================================
# VERIFY SIGNATURE
# =====================================================
@app.route("/verify-signature", methods=["POST"])
def verify_signature():

    try:

        customer_id = request.form.get("customerId", "").strip()

        if not customer_id:
            return jsonify({"success": False, "error": "Customer ID required"}), 400

        customer_data = load_customer_data(customer_id)

        if customer_data is None:
            return jsonify({"success": False, "error": "Customer not found"}), 404

        file = request.files.get("signature")

        if not file or not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Valid signature required"}), 400

        file_bytes = file.read()

        normalized_img, _ = preprocess_signature(
            file_bytes,
            from_bytes=True
        )

        test_emb = get_embedding(normalized_img)

        centroid = customer_data["centroid"]
        mean_d = customer_data["mean_distance"]
        std_d = customer_data["std_distance"]

        # ----------------------------------
        # Adaptive Threshold
        # ----------------------------------
        threshold = mean_d + (2 * std_d)

        distance = float(np.linalg.norm(test_emb - centroid))

        is_genuine = distance < threshold

        if is_genuine:
                # Confidence that the signature is genuine
            confidence = np.exp(-distance / threshold) * 100
        else:
                # Confidence that the signature is forged
            confidence = (1 - np.exp(-distance / threshold)) * 100

        confidence = round(confidence, 2)
        
        return jsonify({
            "success": True,
            "customerId": customer_id,
            "customerName": customer_data["customer_name"],
            "result": "GENUINE" if is_genuine else "FORGED",
            "isGenuine": is_genuine,
            "confidence": confidence,
            "distance": round(distance, 4),
            "threshold": round(threshold, 4),
            "storedSignatures": customer_data["num_signatures"]
        })

    except Exception as e:

        print("\n ERROR in /verify-signature")
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":

    print("\n===================================")
    print("Signify Server Running")
    print("http://127.0.0.1:5000")
    print("===================================\n")

    app.run(debug=True, port=5000)