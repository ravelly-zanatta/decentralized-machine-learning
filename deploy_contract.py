from web3 import Web3
import json
import os
from dotenv import load_dotenv

load_dotenv()

def deploy_contract():
    ganache_url = "http://localhost:8545"              ### http://127.0.0.1:8545
    web3 = Web3(Web3.HTTPProvider(ganache_url))
    account = web3.eth.account.from_key(os.getenv("PRIVATE_KEY"))
    
    with open("contracts/Coordinator.json") as f:
        contract_data = json.load(f)
    
    #abi = contract_data["abi"]
    #contract = web3.eth.contract(address='CONTRACT_ADDRESS', abi=abi)
    contract = web3.eth.contract(abi=contract_data['abi'], bytecode=contract_data['bytecode'])
    tx = contract.constructor(5).build_transaction({
        'from': account.address,
        'nonce': web3.eth.get_transaction_count(account.address, 'latest'),
        'gas': 3000000,
        'gasPrice': web3.to_wei('20', 'gwei')
    })
    
    signed_tx = web3.eth.account.sign_transaction(tx, account._private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    
    print(f"Contract deployed at: {receipt.contractAddress}")
    return receipt.contractAddress

if __name__ == "__main__":
    deploy_contract()
