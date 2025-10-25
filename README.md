Decentralized ML with Hyperledger Besu and IPFS

Este guia assume que você está usando um sistema baseado em Linux (ex.: Ubuntu) ou macOS, mas pode ser adaptado para outros sistemas.

Install Dependencies:
''poetry install''


Setup Hyperledger Besu:

Follow Besu Quickstart to set up a private network with 5 nodes.
Update .env with the private key of the account used to deploy the contract.


Compile and Deploy Contract:

Open contracts/FLCoordinator.sol in Remix IDE.

Compile and export the ABI and bytecode to contracts/FLCoordinator.json.

Run:
python scripts/deploy_contract.py


Update .env with CONTRACT_ADDRESS and CONTRACT_ABI.



Prepare Dataset:

Ensure dataset.csv has columns: Medicamentos, Dose, Frequencia, target.
The dataset will be automatically split into 5 parts.


Run IPFS:

Install and run IPFS daemon:
ipfs daemon




Run Nodes:

Start each node in a separate terminal:
poetry run python src/node.py 0
poetry run python src/node.py 1
poetry run python src/node.py 2
poetry run python src/node.py 3
poetry run python src/node.py 4




Start Rounds:

Manually call startRound() in Remix IDE for each round, or automate via a script.



Environment Variables
Create a .env file:
PRIVATE_KEY=your_private_key
CONTRACT_ADDRESS=deployed_contract_address
CONTRACT_ABI=contract_abi_json

Testing with Hyperledger Caliper

Follow Caliper documentation to set up and benchmark the Besu network.

