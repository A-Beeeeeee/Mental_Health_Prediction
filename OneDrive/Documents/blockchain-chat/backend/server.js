const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const { JsonRpcProvider, Wallet, Contract } = require('ethers');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const mysql = require('mysql2/promise'); 
require('dotenv').config();

// Initialize Express app
const app = express();

// Enable detailed error logging
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send('Something broke!');
});

// Configure CORS
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// Parse JSON request bodies
app.use(express.json());

// Log all requests for debugging
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  next();
});

// Create HTTP server
const server = http.createServer(app);

// Initialize Socket.IO
const io = new Server(server, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST'],
  },
});

// Database connection pool
const pool = mysql.createPool({
  host: process.env.DB_HOST || 'localhost',
  user: process.env.DB_USER || 'root',
  password: process.env.DB_PASSWORD || '',
  database: process.env.DB_NAME || 'blockchain_chat',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

// Function to initialize database tables if they don't exist
async function initDatabase() {
  try {
    const connection = await pool.getConnection();
    console.log('Connected to database');
    
    // Check if users table exists, create if not
    await connection.query(`
      CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    // Check if messages table exists, create if not - WITH blockchain_tx column
    await connection.query(`
      CREATE TABLE IF NOT EXISTS messages (
        id INT AUTO_INCREMENT PRIMARY KEY,
        content TEXT NOT NULL,
        username VARCHAR(50) NOT NULL,
        timestamp BIGINT NOT NULL,
        blockchain_tx VARCHAR(66),
        deleted BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (username) REFERENCES users(username)
      )
    `);
    
    // Check if the deleted column exists, add it if not
    try {
      await connection.query("SELECT deleted FROM messages LIMIT 1");
      console.log("Deleted column exists in messages table");
    } catch (error) {
      if (error.message.includes("Unknown column")) {
        await connection.query("ALTER TABLE messages ADD COLUMN deleted BOOLEAN DEFAULT FALSE");
        console.log("Added deleted column to messages table");
      } else {
        throw error;
      }
    }
    
    // Check if the blockchain_tx column exists, add it if not
    try {
      await connection.query("SELECT blockchain_tx FROM messages LIMIT 1");
      console.log("blockchain_tx column exists in messages table");
    } catch (error) {
      if (error.message.includes("Unknown column")) {
        await connection.query("ALTER TABLE messages ADD COLUMN blockchain_tx VARCHAR(66)");
        console.log("Added blockchain_tx column to messages table");
      } else {
        throw error;
      }
    }
    
    connection.release();
    console.log('Database initialized');
  } catch (error) {
    console.error('Database initialization error:', error);
    process.exit(1);
  }
}

// Load contract ABI
const chatABI = [
  {
    "anonymous": false,
    "inputs": [
      {
        "indexed": true,
        "internalType": "address",
        "name": "sender",
        "type": "address"
      },
      {
        "indexed": false,
        "internalType": "string",
        "name": "content",
        "type": "string"
      },
      {
        "indexed": false,
        "internalType": "uint256",
        "name": "timestamp",
        "type": "uint256"
      }
    ],
    "name": "MessageStored",
    "type": "event"
  },
  {
    "inputs": [],
    "name": "getAllMessages",
    "outputs": [
      {
        "components": [
          {
            "internalType": "address",
            "name": "sender",
            "type": "address"
          },
          {
            "internalType": "string",
            "name": "content",
            "type": "string"
          },
          {
            "internalType": "uint256",
            "name": "timestamp",
            "type": "uint256"
          }
        ],
        "internalType": "struct ChatStorage.Message[]",
        "name": "",
        "type": "tuple[]"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "name": "messages",
    "outputs": [
      {
        "internalType": "address",
        "name": "sender",
        "type": "address"
      },
      {
        "internalType": "string",
        "name": "content",
        "type": "string"
      },
      {
        "internalType": "uint256",
        "name": "timestamp",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "string",
        "name": "content",
        "type": "string"
      },
      {
        "internalType": "uint256",
        "name": "timestamp",
        "type": "uint256"
      }
    ],
    "name": "storeMessage",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  }
];

// Initialize blockchain provider and contract with error handling
let provider, wallet, chatContract;

async function initBlockchain() {
  try {
    // Check if required environment variables are present
    if (!process.env.RPC_URL) {
      console.error('ERROR: RPC_URL environment variable is not defined');
      return false;
    }
    if (!process.env.PRIVATE_KEY) {
      console.error('ERROR: PRIVATE_KEY environment variable is not defined');
      return false;
    }
    if (!process.env.CONTRACT_ADDRESS) {
      console.error('ERROR: CONTRACT_ADDRESS environment variable is not defined');
      return false;
    }

    // Create provider with connection timeout to avoid infinite retries
    console.log(`Attempting to connect to blockchain RPC: ${process.env.RPC_URL}`);
    
    // Adding a timeout to the provider connection attempt
    const connectionPromise = new Promise(async (resolve, reject) => {
      try {
        provider = new JsonRpcProvider(process.env.RPC_URL);
        
        // Test the connection
        await provider.getBlockNumber();
        resolve(true);
      } catch (error) {
        reject(error);
      }
    });
    
    // Set a timeout for the connection attempt
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => reject(new Error("Blockchain connection timeout after 5 seconds")), 5000);
    });
    
    // Race between connection and timeout
    const connected = await Promise.race([connectionPromise, timeoutPromise])
      .catch(error => {
        console.error('Blockchain connection failed:', error.message);
        return false;
      });
      
    if (!connected) {
      return false;
    }
    
    console.log('Connected to blockchain provider');
    
    wallet = new Wallet(process.env.PRIVATE_KEY, provider);
    
    // Check wallet balance
    const balance = await provider.getBalance(wallet.address);
    console.log(`Wallet ${wallet.address} balance: ${balance.toString()}`);
    if (balance.toString() === '0') {
      console.warn('WARNING: Wallet has zero balance, may not be able to send transactions');
    }
    
    chatContract = new Contract(
      process.env.CONTRACT_ADDRESS,
      chatABI,
      wallet
    );
    
    // Test if the contract is accessible
    try {
      // Call a read-only method to verify contract connection
      await chatContract.getAllMessages();
      console.log('Successfully connected to chat contract');
      return true;
    } catch (error) {
      console.error('Failed to connect to chat contract:', error.message);
      // Don't fail completely if contract method fails but contract exists
      // This could be because the contract is deployed but getAllMessages returns nothing
      return true;
    }
  } catch (error) {
    console.error('Blockchain initialization error:', error);
    return false;
  }
}

// Flag to track blockchain status
let blockchainEnabled = false;

// Test route to verify server is running
app.get('/api/test', (req, res) => {
  res.json({ 
    message: 'Server is running!',
    blockchainEnabled: blockchainEnabled
  });
});

// Authentication middleware for API routes
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) return res.status(401).json({ error: 'Access denied' });

  jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
    if (err) return res.status(403).json({ error: 'Invalid token' });
    req.user = user;
    next();
  });
};

// Authentication middleware for WebSocket
const authenticateSocketConnection = (socket, next) => {
  const token = socket.handshake.auth.token;
  if (!token) {
    return next(new Error('Authentication error'));
  }

  jwt.verify(token, process.env.JWT_SECRET, (err, decoded) => {
    if (err) {
      return next(new Error('Authentication error'));
    }
    socket.user = decoded;
    next();
  });
};

// Apply WebSocket authentication
io.use(authenticateSocketConnection);

// Register endpoint with SQL database
app.post('/api/register', async (req, res) => {
  try {
    console.log('Registration attempt:', req.body);
    const { username, password } = req.body;
    
    // Validate input
    if (!username || !password) {
      console.log('Missing username or password');
      return res.status(400).json({ error: 'Username and password are required' });
    }
    
    // Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);
    
    // Store new user in database
    const [result] = await pool.query(
      'INSERT INTO users (username, password) VALUES (?, ?)',
      [username, hashedPassword]
    );
    
    console.log('User registered successfully:', username);
    res.status(201).json({ message: 'User registered successfully' });
  } catch (error) {
    console.error('Registration error:', error);
    
    // Handle duplicate username
    if (error.code === 'ER_DUP_ENTRY') {
      return res.status(400).json({ error: 'Username already exists' });
    }
    
    res.status(500).json({ error: 'Internal server error: ' + error.message });
  }
});

// Login endpoint with SQL database
app.post('/api/login', async (req, res) => {
  try {
    console.log('Login attempt:', req.body.username);
    const { username, password } = req.body;
    
    // Find user in database
    const [users] = await pool.query(
      'SELECT * FROM users WHERE username = ?',
      [username]
    );
    
    if (users.length === 0) {
      console.log('User not found:', username);
      return res.status(400).json({ error: 'Invalid username or password' });
    }
    
    const user = users[0];
    
    // Validate password
    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) {
      console.log('Invalid password for user:', username);
      return res.status(400).json({ error: 'Invalid username or password' });
    }
    
    // Create and send JWT
    const token = jwt.sign(
      { id: user.id, username: user.username },
      process.env.JWT_SECRET,
      { expiresIn: '24h' }
    );
    
    console.log('Login successful:', username);
    res.json({ 
      token, 
      username: user.username,
      blockchainEnabled: blockchainEnabled // Let clients know if blockchain is enabled
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Internal server error: ' + error.message });
  }
});

// Get messages from database - updated to exclude deleted messages
app.get('/api/messages', authenticateToken, async (req, res) => {
  try {
    console.log('Fetching messages for user:', req.user.username);
    
    // Get non-deleted messages from database
    const [messages] = await pool.query(
      'SELECT * FROM messages WHERE deleted = FALSE ORDER BY timestamp ASC'
    );
    
    res.json({ messages });
  } catch (error) {
    console.error('Error fetching messages:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Delete message endpoint
app.delete('/api/messages/:id', authenticateToken, async (req, res) => {
  try {
    const messageId = req.params.id;
    const username = req.user.username;
    
    console.log(`Delete request for message ${messageId} by user ${username}`);
    
    // First, verify that the user owns this message
    const [messages] = await pool.query(
      'SELECT * FROM messages WHERE id = ? AND username = ?',
      [messageId, username]
    );
    
    if (messages.length === 0) {
      return res.status(403).json({ 
        error: 'Not authorized to delete this message or message not found' 
      });
    }
    
    // Soft delete by setting deleted flag
    await pool.query(
      'UPDATE messages SET deleted = TRUE WHERE id = ?',
      [messageId]
    );
    
    console.log(`Message ${messageId} marked as deleted`);
    
    // Broadcast the deletion to all connected clients
    io.emit('message-deleted', { id: messageId });
    
    res.json({ message: 'Message deleted successfully' });
  } catch (error) {
    console.error('Error deleting message:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Socket connection handler with improved error handling
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  console.log('Authenticated user:', socket.user.username);
  
  // Inform client about blockchain status
  socket.emit('blockchain-status', { enabled: blockchainEnabled });
  
  socket.on('send-message', async (data) => {
    console.log('Received message from', socket.user.username, ':', data);
    
    try {
      // Add user info to the message
      const messageWithUser = {
        ...data,
        username: socket.user.username
      };
      
      let txHash = null;
      
      // Try to store on blockchain if enabled
      if (blockchainEnabled) {
        try {
          console.log(`Storing message on blockchain: "${data.content}" with timestamp ${data.timestamp}`);
          const tx = await chatContract.storeMessage(data.content, data.timestamp);
          console.log('Message submitted to blockchain, transaction hash:', tx.hash);
          
          txHash = tx.hash;
          
          // Wait for transaction confirmation
          const receipt = await tx.wait(1);
          console.log('Blockchain transaction confirmed in block:', receipt.blockNumber);
        } catch (blockchainError) {
          console.error('Blockchain storage error:', blockchainError);
          // Continue with database storage, but note the blockchain error
          console.log('Continuing with database storage despite blockchain error');
        }
      }
      
      // Store in database (even if blockchain failed)
      const [result] = await pool.query(
        'INSERT INTO messages (content, username, timestamp, blockchain_tx) VALUES (?, ?, ?, ?)',
        [data.content, socket.user.username, data.timestamp, txHash]
      );
      
      // Get the ID of the inserted message
      const messageId = result.insertId;
      
      // Broadcast to all clients with the message ID
      io.emit('receive-message', {
        id: messageId,
        ...messageWithUser,
        blockchain_tx: txHash
      });
    } catch (error) {
      console.error('Error storing message:', error);
      socket.emit('error', { message: 'Failed to store message: ' + error.message });
    }
  });
  
  // Handle message deletion through socket
  socket.on('delete-message', async (data) => {
    try {
      const messageId = data.id;
      const username = socket.user.username;
      
      console.log(`Socket delete request for message ${messageId} by user ${username}`);
      
      // First, verify that the user owns this message
      const [messages] = await pool.query(
        'SELECT * FROM messages WHERE id = ? AND username = ?',
        [messageId, username]
      );
      
      if (messages.length === 0) {
        socket.emit('error', { 
          message: 'Not authorized to delete this message or message not found' 
        });
        return;
      }
      
      // Soft delete by setting deleted flag
      await pool.query(
        'UPDATE messages SET deleted = TRUE WHERE id = ?',
        [messageId]
      );
      
      console.log(`Message ${messageId} marked as deleted`);
      
      // Broadcast the deletion to all connected clients
      io.emit('message-deleted', { id: messageId });
      
      socket.emit('delete-success', { id: messageId });
    } catch (error) {
      console.error('Error deleting message:', error);
      socket.emit('error', { message: 'Failed to delete message' });
    }
  });
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Start the server
const PORT = process.env.PORT || 5000;
(async () => {
  await initDatabase();
  
  // Initialize blockchain connection
  blockchainEnabled = await initBlockchain();
  
  server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`JWT Secret defined: ${Boolean(process.env.JWT_SECRET)}`);
    console.log(`Blockchain integration: ${blockchainEnabled ? 'ENABLED' : 'DISABLED'}`);
    console.log(`Database connection established`);
  });
})();