// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Coordinator {
    address public owner;
    uint256 public currentRound;
    uint256 public totalRounds;
    address[] public nodes;
    mapping(address => bool) public registeredNodes;
    mapping(uint256 => address) public aggregators;
    mapping(uint256 => address) public testers;
    mapping(uint256 => string[]) public trainingCIDs; // CIDs de treinamento
    mapping(uint256 => string) public aggregatedCID; // CID agregado (único por round)
    mapping(uint256 => string) public testCID; // CID de teste (parâmetros agregados + métricas)
    mapping(uint256 => mapping(address => bool)) public hasStoredTrainingCID; // Rastreia CIDs de treinamento

    event NodeRegistered(address node);
    event RoundStarted(uint256 indexed round, address aggregator, address tester);
    event TrainingCIDStored(uint256 indexed round, address node, string cid);
    event AggregatedCIDStored(uint256 indexed round, address aggregator, string cid);
    event TestCIDStored(uint256 indexed round, address tester, string cid);
    event RoundCompleted(uint256 indexed round);

    constructor(uint256 _totalRounds) {
        owner = msg.sender;
        currentRound = 0;
        totalRounds = _totalRounds;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this");
        _;
    }

    modifier onlyRegistered() {
        require(registeredNodes[msg.sender], "Node not registered");
        _;
    }

    function registerNode(address _node) external onlyOwner {
        require(!registeredNodes[_node], "Node already registered");
        registeredNodes[_node] = true;
        nodes.push(_node);
        emit NodeRegistered(_node);

        // Iniciar o primeiro round automaticamente quando 4 nós forem registrados
        if (nodes.length == 4 && currentRound == 0) {
            startRound();
        }
    }

    function storeTrainingCID(string memory _cid) external onlyRegistered {
        require(currentRound > 0, "No round has started");
        require(currentRound <= totalRounds, "Training has completed");
        require(msg.sender != testers[currentRound], "Tester cannot store training CID");
        require(!hasStoredTrainingCID[currentRound][msg.sender], "Training CID already stored");

        hasStoredTrainingCID[currentRound][msg.sender] = true;
        trainingCIDs[currentRound].push(_cid);
        emit TrainingCIDStored(currentRound, msg.sender, _cid);
    }

    function storeAggregatedCID(string memory _aggregatedCid) external {
        require(currentRound > 0, "No round has started");
        require(msg.sender == aggregators[currentRound], "Only the aggregator can store aggregated CID");
        require(hasStoredTrainingCID[currentRound][msg.sender], "Aggregator must store training CID first");
        // Verificar se todos os nós (exceto o testador) enviaram CIDs
        uint256 requiredCIDs = nodes.length - 1; // Todos os nós menos o testador
        require(trainingCIDs[currentRound].length >= requiredCIDs, "Not all training CIDs stored");

        aggregatedCID[currentRound] = _aggregatedCid;
        emit AggregatedCIDStored(currentRound, msg.sender, _aggregatedCid);
    }

    function completeRound(string memory _testCid) external {
        require(currentRound > 0, "No round has started");
        require(msg.sender == testers[currentRound], "Only the tester can complete the round");
        require(bytes(aggregatedCID[currentRound]).length > 0, "Aggregated CID not stored yet");

        testCID[currentRound] = _testCid;
        emit TestCIDStored(currentRound, msg.sender, _testCid);
        emit RoundCompleted(currentRound);

        if (currentRound < totalRounds) {
            startRound();
        }
    }

    function startRound() private {
        require(nodes.length >= 4, "At least four nodes must be registered");
        require(currentRound < totalRounds, "All rounds completed");

        currentRound++;

        // Pseudo-random selection
        uint256 seed = uint256(blockhash(block.number - 1));
        address aggregator = nodes[seed % nodes.length];
        address tester;
        do {
            seed = uint256(keccak256(abi.encodePacked(seed)));
            tester = nodes[seed % nodes.length];
        } while (tester == aggregator);

        aggregators[currentRound] = aggregator;
        testers[currentRound] = tester;

        // Resetar hasStoredTrainingCID para todos os nós no novo round
        for (uint256 i = 0; i < nodes.length; i++) {
            hasStoredTrainingCID[currentRound][nodes[i]] = false;
        }
        emit RoundStarted(currentRound, aggregator, tester);
    }

    function getTrainingCIDs(uint256 _round) external view returns (string[] memory) {
        return trainingCIDs[_round];
    }

    function getAggregatedCID(uint256 _round) external view returns (string memory) {
        return aggregatedCID[_round];
    }

    function getTestCID(uint256 _round) external view returns (string memory) {
        return testCID[_round];
    }

    function getAggregator(uint256 _round) external view returns (address) {
        return aggregators[_round];
    }

    function getTester(uint256 _round) external view returns (address) {
        return testers[_round];
    }

    function getCurrentRound() external view returns (uint256) {
        return currentRound;
    }

    function getNumberOfNodes() external view returns (uint256) {
        return nodes.length;
    }
}
