# Decentralized ML with Hyperledger Besu and IPFS

This guide assumes that you are using a Linux-based system (e.g., Ubuntu) or macOS, but it can be adapted for other operating systems.

## 1. Setup the Environment
### Install Dependencies:
- Install Python 3.9+:
  ```
  sudo apt update
  sudo apt install python3.9 python3.9-venv
  ```
- Install Poetry:
    - Install Poetry to manage dependencies:
    ```
    curl -sSL https://install.python-poetry.org | python3 -
    ```
    - Add Poetry to the PATH (if necessary):
    ```
    export PATH="$HOME/.local/bin:$PATH"
    ```
- Install IPFS:
  ```
  sudo snap install ipfs
  ```
- Install Hyperledger Besu:
  ```
  wget https://binaries.hyperledger.org/besu/latest/besu-latest.zip
  unzip besu-latest.zip -d besu
  ```
- Install Python Dependencies:
  ```
  poetry install
  ```

## 2. Network Settings:
### Setup Hyperledger Besu:
- Setup a private network with 5 nodes using Quickstart:
  ```
  git clone https://github.com/hyperledger/besu
  cd besu/docs/examples/quickstart
  ./run.sh
  ```
### Setup the Smart Contract:
- Open the `contracts/Coordinator.sol` file in Remix IDE.
- Compile the contract.
- Export the ABI and Bytecode to `contracts/Coordinator.json`.
  ```
  {
  "abi": [...],
  "bytecode": "0x..."
  }
  ```
- Setup the local network in MetaMask.
- Add the Besu node in MetaMask using `PRIVATE_KEY`.
- Connect the MetaMask wallet in Remix IDE (**Injected Provider - MetaMask**)
  
### Create `.env` File:
- Create the `.env` file with:
  ```
  PRIVATE_KEY = 0x...
  PRIVATE_KEY_0 = 0x...
  PRIVATE_KEY_1 = 0x...
  PRIVATE_KEY_2 = 0x...
  PRIVATE_KEY_3 = 0x...
  NODE_ADDRESS = 0x...
  NODE_ADDRESS_0 = 0x...
  NODE_ADDRESS_1 = 0x...
  NODE_ADDRESS_2 = 0x...
  NODE_ADDRESS_3 = 0x...
  CONTRACT_ADDRESS = 0x...
  CONTRACT_ABI = [...]
  WEB3_PROVIDER = http://localhost:8545
  ```
    - **PRIVATE_KEY**: Private key of the Besu nodes.
    - **NODE_ADDRESS**: Address of the Besu nodes.
    - **CONTRACT_ADDRESS**: Fill in after deploying the contract.
    - **CONTRACT_ABI**: Copy the ABI from Remix IDE and paste it here.
 
## Run Training:
- Deploy the contract in Remix IDE.
- Paste the contract address into `CONTRACT_ADDRESS` in the `.env` file.
- Start IPFS:
  ```
  IPFS daemon
  ```
- Start the nodes:
    - In separate terminals, run each node:
      ```
      poetry run python src/node.py 0
      poetry run python src/node.py 1
      poetry run python src/node.py 2
      poetry run python src/node.py 3
      ```
- Register the nodes by calling the contract function `registerNode`.     



