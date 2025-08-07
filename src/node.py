import os
import time
import json
from web3 import Web3
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from utils import load_and_split_dataset, get_model_parameters, set_model_params, set_initial_params, aggregate_parameters, set_model_params_test
from ipfs_utils import save_to_ipfs, load_from_ipfs

load_dotenv()

class Node:
    def __init__(self, node_id, dataset_path, contract_address, web3_provider):
        self.node_id = node_id
        self.dataset_path = dataset_path
        #self.model = DecisionTreeClassifier(random_state=42)
        self.model = LogisticRegression(random_state=42, max_iter=1000, warm_start=True)
        # self.X_train, self.X_test, self.y_train, self.y_test = load_and_split_dataset(dataset_path, num_nodes=3, node_id=node_id)
        self.web3 = Web3(Web3.HTTPProvider(web3_provider))
        if not self.web3.is_connected():
            raise Exception(f"Node {node_id} failed to connect to Besu at {web3_provider}")
        print(f"Node {node_id} connected to Besu. Block number: {self.web3.eth.block_number}")
        
        private_key_env_name = f"PRIVATE_KEY_{node_id}"
        private_key = os.getenv(private_key_env_name)
        self.account = self.web3.eth.account.from_key(private_key)
        #self.account = self.web3.eth.account.from_key(os.getenv('PRIVATE_KEY'))
        with open("contracts/SLCoordinator.json") as f:
            contract_data = json.load(f)
        abi = contract_data["abi"]
        self.contract = self.web3.eth.contract(address=contract_address, abi=abi)
        #self.contract = self.web3.eth.contract(address=contract_address, abi=json.loads(os.getenv('CONTRACT_ABI')))
        
        set_initial_params(self.model)
        
    def load_dataset(self, is_tester=False, trainer_position=-1):
        number_of_nodes = self.contract.functions.getNumberOfNodes().call()
        print(f"Numero de nodes: {number_of_nodes}")
        self.X_node_train, self.X_node_test, self.X_test, self.y_node_train, self.y_node_test, self.y_test = load_and_split_dataset(self.dataset_path, num_nodes=number_of_nodes, node_id=self.node_id, is_tester=is_tester, trainer_position=trainer_position)
    
    def train(self, round_number):
        try:
            if round_number != 1:
                cid = self.contract.functions.getTestCID(round_number - 1).call()
                data = load_from_ipfs(cid)
                self.model = set_model_params_test(self.model, data['params'])
            self.model.fit(self.X_node_train, self.y_node_train)
            loss = log_loss(self.y_node_test, self.model.predict_proba(self.X_node_test))
            print(f"Loss: {loss}")
            accuracy = self.model.score(self.X_node_test, self.y_node_test)
            print(f"Accuracy: {accuracy}")
            y_pred = self.model.predict(self.X_node_test)
            precision = precision_score(self.y_node_test, y_pred, average='binary')
            print(f"Precision: {precision}")
            recall = recall_score(self.y_node_test, y_pred, average='binary')
            print(f"Recall: {recall}")
            f1 = f1_score(self.y_node_test, y_pred, average='binary')
            print(f"F1: {f1}")
            params = get_model_parameters(self.model)
            data = {'params': params, 'metrics': {'loss': loss, 'accuracy': accuracy, 'precision':precision, 'recall':recall, 'f1':f1}, 'node_id': self.node_id}
            #ipfs_start = time.time()
            cid = save_to_ipfs(data)
            #ipfs_end = time.time()
            #print(f"CID: {cid} - Saved to IPFS in {ipfs_end - ipfs_start} seconds")
            self._send_transaction(self.contract.functions.storeTrainingCID, cid)
            return cid
        except Exception as e:
            print(f"Node {self.node_id} - Error in train: {str(e)}")
            raise e
    
    def aggregate(self, round_number):
        try:
            # Esperar todos os CIDs de treinamento
            required_cids = self.contract.functions.getNumberOfNodes().call() - 1
            cids = []
            timeout = 300
            start_time = time.time()
            while len(cids) < required_cids:
                cids = self.contract.functions.getTrainingCIDs(round_number).call()
                print(f"Node {self.node_id} - Retrieved {len(cids)}/{required_cids} training CIDs for round {round_number}: {cids}")
                if time.time() - start_time > timeout:
                    raise Exception(f"Node {self.node_id} - Timeout waiting for training CIDs in round {round_number}")
                time.sleep(3)
            
            params_list = []
            for cid in cids:
                data = load_from_ipfs(cid)
                params_list.append(data['params'])
            aggregated_params = aggregate_parameters(params_list)
            #ipfs_start = time.time()
            cid = save_to_ipfs({'params': aggregated_params, 'node_id': self.node_id})
            #ipfs_end = time.time()
            #print(f"CID: {cid} - Saved to IPFS in {ipfs_end - ipfs_start} seconds")
            self._send_transaction(self.contract.functions.storeAggregatedCID, cid)
            print(f"Node {self.node_id} - Sent aggregated CID: {cid} for round {round_number}")
            return cid
        except Exception as e:
            print(f"Node {self.node_id} - Error in Aggregate: {str(e)}")
            raise e
    
    def test(self, round_number):
        try:
            # Esperar o CID agregado
            cid = ""
            timeout = 300
            start_time = time.time()
            while not cid:
                cid = self.contract.functions.getAggregatedCID(round_number).call()
                print(f"Node {self.node_id} - Checking aggregated CID for round {round_number}: {cid}")
                if time.time() - start_time > timeout:
                    raise Exception(f"Node {self.node_id} - Timeout waiting for aggregated CID in round {round_number}")
                time.sleep(3)
            
            data = load_from_ipfs(cid)
            print(f"DADOS PARAMETROS AGREGADOS: {data}")
            self.model = set_model_params_test(self.model, data['params'])
            loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
            y_pred = self.model.predict(self.X_test)
            accuracy = self.model.score(self.X_test, self.y_test)
            precision = precision_score(self.y_test, y_pred, average='binary')
            recall = recall_score(self.y_test, y_pred, average='binary')
            f1 = f1_score(self.y_test, y_pred, average='binary')
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            metrics = {'params': data['params'], 'metrics': {'loss': float(loss), 'accuracy': float(accuracy), 'precision': precision, 'recall': recall, 'f1': f1}, 'node_id': self.node_id}
            #ipfs_start = time.time()
            print(f"Métricas finais do round: {metrics['metrics']}")
            print(f"Matrix de confusão: {conf_matrix}")
            test_cid = save_to_ipfs(metrics)
            #ipfs_end = time.time()
            #print(f"CID: {cid} - Saved to IPFS in {ipfs_end - ipfs_start} seconds")
            self._send_transaction(self.contract.functions.completeRound, test_cid)
            print(f"Node {self.node_id} - Sent test CID: {test_cid} for round {round_number}")
            return test_cid
        except Exception as e:
            print(f"Node {self.node_id} - Error in Test: {str(e)}")
            raise e
    
    def _send_transaction(self, function, *args):
        try:
            tx = function(*args).build_transaction({
                'from': self.account.address,
                'nonce': self.web3.eth.get_transaction_count(self.account.address, 'pending'),
                'gas': 2000000,
                'gasPrice': self.web3.to_wei('30', 'gwei')               # mudar para 20 gwei novamente
            })
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.account._private_key)
            tx_start = time.time()
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            tx_end = time.time()
            print(f"================================== LATÊNCIAAAAA TRANSAÇÃO: {tx_end - tx_start} segundos, Hash: {tx_hash.hex()} ======================================")
            return receipt
        except Exception as e:
            print(f"Node {self.node_id} - Error sending transaction: {str(e)}")
            raise e
    
    def run(self):
        try:
            total_rounds = self.contract.functions.totalRounds().call()
            for round_number in range(1, total_rounds + 1):              
                print(f"Node {self.node_id} - Waiting for round {round_number} to start...")   
                timeout = 300
                start_time = time.time()
                while True:
                    try:
                        current_round = self.contract.functions.getCurrentRound().call()
                        aggregator = self.contract.functions.getAggregator(round_number).call()
                        #print(f"Node {self.node_id} - Current round: {current_round}, Aggregator for round {round_number}: {aggregator}")
                        if aggregator != '0x0000000000000000000000000000000000000000' and current_round >= round_number:
                            break
                        if time.time() - start_time > timeout:
                            raise Exception(f"Node {self.node_id} - Timeout waiting for round {round_number} to start")
                        time.sleep(5)
                    except Exception as e:
                        print(f"Node {self.node_id} - Error checking aggregator for round {round_number}: {str(e)}")
                        time.sleep(5)
                print(f"Node {self.node_id} - Round {round_number} started.")
                round_start = time.time()

                tester = self.contract.functions.getTester(round_number).call()
                print(f"Node {self.node_id} - Tester for round {round_number}: {tester}")
                
                node_list = self.contract.functions.getNodeAddresses().call()
                print("Node list: ", node_list)
                node_list.remove(tester)

                # Fase de treinamento (exceto para o testador)
                if self.web3.to_checksum_address(self.account.address) != tester:
                    # Load dataset
                    trainer_position = node_list.index(self.account.address)
                    self.load_dataset(is_tester=False, trainer_position=trainer_position)
                    print(f"Node {self.node_id} - Training for round {round_number}...")
                    self.train(round_number)

                # Fase de agregação (apenas para o agregador)
                if self.web3.to_checksum_address(self.account.address) == aggregator:
                    print(f"Node {self.node_id} is aggregator for round {round_number}")
                    #cid = self.aggregate(round_number)
                    self.aggregate(round_number)

                # Fase de teste (apenas para o testador)
                if self.web3.to_checksum_address(self.account.address) == tester:
                    # Load dataset
                    self.load_dataset(is_tester=True, trainer_position=-1)
                    print(f"Node {self.node_id} is tester for round {round_number}")
                    self.test(round_number)

                print(f"Node {self.node_id} - Completed round {round_number}")
                round_end = time.time()
                print(f"########################################-Node {self.node_id} - Round {round_number} DURATION: {round_end - round_start} SECONDS-##########################################3")
                time.sleep(10)
        except Exception as e:
            print(f"Node {self.node_id} - Error in Run: {str(e)}")
            raise e


if __name__ == "__main__":
    import sys
    node_id = int(sys.argv[1])
    node = Node(
        node_id=node_id,
        dataset_path="dataset.csv",
        contract_address=os.getenv("CONTRACT_ADDRESS"),
        web3_provider="http://localhost:8545"
    )
    node.run()