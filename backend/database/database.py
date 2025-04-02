import firebase_admin
from firebase_admin import credentials, firestore

# Initialize the Firebase app with the service account credentials
cred = credentials.Certificate("database/thesis-db-23474-firebase-adminsdk-fbsvc-d1ecfa0eb4.json")
firebase_admin.initialize_app(cred)

# Get a Firestore client
db = firestore.client()

# Create a Firestore collection reference
collection_ref = db.collection("10days_prediction")

def add_data_to_firestore(data: dict):
    """
    Add data to Firestore.
    :param data: Dictionary containing the data to be added.
    sample data:
    {
        "date": "2023-10-01",
        "predicted_10days_data": [
                123.45,
                124.56,
                125.67,
                126.78,
                127.89,
                128.90,
                129.01,
                130.12,
                131.23,
                132.34,
        ]
    }
    """
    try:
        # Add a new document with a generated ID
        doc_ref = collection_ref.add(data)
        print(f"Document added with ID: {doc_ref[1].id}")
    except Exception as e:
        print(f"Error adding document: {e}")

#get data from Firestore from date
def get_data_from_firestore(date: str):
    """
    Get data from Firestore by date.
    :param date: Date string in the format 'YYYY-MM-DD'.
    :return: Document data if found, None otherwise.
    """
    try:
        # Query the collection for documents with the specified date
        query = collection_ref.where("date", "==", date).get()
        if query:
            return query[0].to_dict()
        else:
            print(f"No document found for date: {date}")
            return None
    except Exception as e:
        print(f"Error fetching document: {e}")
        return None

    
if __name__ == "__main__":
    # Example usage
    data = {
        "date": "2023-10-01",
        "predicted_10days_data": [
            123.45,
            124.56,
            125.67,
            126.78,
            127.89,
            128.90,
            129.01,
            130.12,
            131.23,
            132.34,
        ]
    }
    
    add_data_to_firestore(data)
    fetched_data = get_data_from_firestore("2023-10-01")
    print(fetched_data)