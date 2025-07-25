// scripts/deploy.js

async function main() {
    const ChatStorage = await ethers.getContractFactory("ChatStorage");
    const chatStorage = await ChatStorage.deploy();
    await chatStorage.deployed();
  
    console.log("ChatStorage deployed to:", chatStorage.address);
  }
  
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
  