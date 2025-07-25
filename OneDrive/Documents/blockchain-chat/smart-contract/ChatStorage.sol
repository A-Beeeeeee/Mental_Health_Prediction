// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ChatStorage {
    struct Message {
        address sender;
        string content;
        uint256 timestamp;
    }

    Message[] public messages;

    event MessageStored(address indexed sender, string content, uint256 timestamp);

    function storeMessage(string memory content, uint256 timestamp) public {
        messages.push(Message(msg.sender, content, timestamp));
        emit MessageStored(msg.sender, content, timestamp);
    }

    function getAllMessages() public view returns (Message[] memory) {
        return messages;
    }
}