import ipfshttpclient
import pickle

def save_to_ipfs(data):
    with ipfshttpclient.connect() as client:
        serialized_data = pickle.dumps(data)
        result = client.add_bytes(serialized_data)
        return result

def load_from_ipfs(cid):
    with ipfshttpclient.connect() as client:
        data = client.cat(cid)
        return pickle.loads(data)