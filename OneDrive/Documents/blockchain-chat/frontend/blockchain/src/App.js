import React, { useState, useEffect } from "react";
import io from "socket.io-client";
import axios from "axios";

// Base URL - change this to match your backend URL
const API_BASE_URL = "http://localhost:5000";

// Common emojis to choose from
const commonEmojis = ["😊", "😂", "❤️", "👍", "🎉", "🔥", "😎", "🙏", "✨", "🤔"];

// Auth components
const AuthForm = ({ onLogin }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");
  const [serverStatus, setServerStatus] = useState(null);

  // Check if server is running on component mount
  useEffect(() => {
    const checkServer = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/test`);
        setServerStatus({
          online: true,
          blockchainEnabled: response.data.blockchainEnabled
        });
      } catch (error) {
        console.error("Server connection error:", error);
        setServerStatus({ online: false });
      }
    };

    checkServer();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccessMessage("");
    setLoading(true);

    try {
      const endpoint = isLogin ? "/api/login" : "/api/register";
      
      if (!isLogin) {
        // Handle registration
        await axios.post(`${API_BASE_URL}${endpoint}`, {
          username,
          password,
        });
        
        setSuccessMessage("Registration successful! You can now log in.");
        setIsLogin(true);
        setPassword("");
      } else {
        // Handle login
        const response = await axios.post(`${API_BASE_URL}${endpoint}`, {
          username,
          password,
        });
        
        localStorage.setItem("token", response.data.token);
        localStorage.setItem("username", response.data.username);
        
        // Also store blockchain status in localStorage
        if (response.data.blockchainEnabled !== undefined) {
          localStorage.setItem("blockchainEnabled", 
            response.data.blockchainEnabled.toString());
        }
        
        onLogin(response.data);
      }
    } catch (error) {
      console.error("Auth error:", error);
      
      if (error.response && error.response.status === 400) {
        setError(error.response.data.error || "Invalid username or password");
      } else if (!error.response) {
        setError("Server connection error. Please check if the server is running.");
      } else {
        setError(
          error.response?.data?.error || 
          "An unexpected error occurred. Please try again."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: "400px", margin: "0 auto", padding: "20px" }}>
      <h2>{isLogin ? "Login" : "Register"}</h2>
      
      {serverStatus && !serverStatus.online && (
        <div style={{ 
          backgroundColor: "#ffebee", 
          color: "#c62828", 
          padding: "10px", 
          borderRadius: "4px",
          marginBottom: "15px" 
        }}>
          <strong>Warning:</strong> The server appears to be offline. Please make sure the backend server is running.
        </div>
      )}
      
      {serverStatus && serverStatus.online && !serverStatus.blockchainEnabled && (
        <div style={{ 
          backgroundColor: "#fff3e0", 
          color: "#e65100", 
          padding: "10px", 
          borderRadius: "4px",
          marginBottom: "15px" 
        }}>
          <strong>Notice:</strong> The blockchain integration is currently disabled. Messages will be stored in the database only.
        </div>
      )}
      
      {error && <p style={{ color: "red" }}>{error}</p>}
      {successMessage && <p style={{ color: "green" }}>{successMessage}</p>}
      
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: "15px" }}>
          <label htmlFor="username">Username</label>
          <input
            type="text"
            id="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            style={{ display: "block", width: "100%", padding: "8px" }}
            required
          />
        </div>
        <div style={{ marginBottom: "15px" }}>
          <label htmlFor="password">Password</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{ display: "block", width: "100%", padding: "8px" }}
            required
          />
        </div>
        <button
          type="submit"
          disabled={loading || (serverStatus && !serverStatus.online)}
          style={{
            padding: "10px 15px",
            backgroundColor: "#4CAF50",
            color: "white",
            border: "none",
            cursor: (loading || (serverStatus && !serverStatus.online)) ? "not-allowed" : "pointer",
            opacity: (loading || (serverStatus && !serverStatus.online)) ? 0.7 : 1
          }}
        >
          {loading
            ? "Loading..."
            : isLogin
            ? "Login"
            : "Create Account"}
        </button>
      </form>
      <p style={{ marginTop: "15px" }}>
        {isLogin ? "Don't have an account? " : "Already have an account? "}
        <button
          onClick={() => {
            setIsLogin(!isLogin);
            setError("");
            setSuccessMessage("");
          }}
          style={{
            background: "none",
            border: "none",
            color: "#4CAF50",
            textDecoration: "underline",
            cursor: "pointer",
          }}
        >
          {isLogin ? "Register" : "Login"}
        </button>
      </p>
    </div>
  );
};

// Message component
const Message = ({ msg, currentUser, onDelete }) => {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const isOwnMessage = msg.username === currentUser;

  const handleDeleteClick = () => {
    setShowDeleteConfirm(true);
  };

  const confirmDelete = () => {
    onDelete(msg.id);
    setShowDeleteConfirm(false);
  };

  const cancelDelete = () => {
    setShowDeleteConfirm(false);
  };

  return (
    <div
      style={{
        marginBottom: "15px",
        textAlign: isOwnMessage ? "right" : "left",
      }}
    >
      <div
        style={{
          display: "inline-block",
          backgroundColor: isOwnMessage ? "#dcf8c6" : "#f1f0f0",
          borderRadius: "12px",
          padding: "10px 15px",
          maxWidth: "70%",
          boxShadow: "0 1px 2px rgba(0,0,0,0.1)",
          position: "relative"
        }}
      >
        {!isOwnMessage && (
          <strong style={{ display: "block", marginBottom: "5px" }}>
            {msg.username}
          </strong>
        )}
        <span>{msg.content}</span>
        <span
          style={{ display: "block", fontSize: "0.7rem", color: "#888", marginTop: "5px" }}
        >
          {new Date(msg.timestamp).toLocaleTimeString()}
          {msg.sending && " (sending...)"}
          {msg.blockchain_tx && (
            <span style={{ marginLeft: "5px", color: "#4CAF50" }}>✓ On blockchain</span>
          )}
        </span>
        
        {/* Delete options - only show for your own messages */}
        {isOwnMessage && !msg.sending && !showDeleteConfirm && (
          <button
            onClick={handleDeleteClick}
            style={{
              background: "none",
              border: "none",
              color: "#f44336",
              cursor: "pointer",
              fontSize: "0.8rem",
              padding: "2px 5px",
              marginTop: "5px",
              display: "block",
              marginLeft: "auto",
              opacity: 0.7
            }}
          >
            Delete
          </button>
        )}
        
        {/* Delete confirmation */}
        {showDeleteConfirm && (
          <div style={{ 
            marginTop: "8px", 
            borderTop: "1px solid #ddd", 
            paddingTop: "8px" 
          }}>
            <p style={{ 
              fontSize: "0.8rem", 
              margin: "0 0 5px 0", 
              fontStyle: "italic" 
            }}>
              Delete this message? {msg.blockchain_tx ? "It will be hidden but still stored on the blockchain." : ""}
            </p>
            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <button
                onClick={cancelDelete}
                style={{
                  background: "none",
                  border: "1px solid #ccc",
                  borderRadius: "4px",
                  padding: "2px 8px",
                  marginRight: "8px",
                  fontSize: "0.8rem",
                  cursor: "pointer"
                }}
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                style={{
                  background: "#f44336",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  padding: "2px 8px",
                  fontSize: "0.8rem",
                  cursor: "pointer"
                }}
              >
                Delete
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Main App component
function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [auth, setAuth] = useState({
    isAuthenticated: false,
    token: null,
    username: null,
  });
  const [socket, setSocket] = useState(null);
  const [showEmojis, setShowEmojis] = useState(false);
  const [isMessageSending, setIsMessageSending] = useState(false);
  const [sendError, setSendError] = useState("");
  const [deleteError, setDeleteError] = useState("");
  const [blockchainEnabled, setBlockchainEnabled] = useState(false);
  const [socketConnected, setSocketConnected] = useState(false);
  const [messagesEndRef] = useState(React.createRef());

  // Scroll function - auto scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Check for existing token on app load
  useEffect(() => {
    const token = localStorage.getItem("token");
    const username = localStorage.getItem("username");
    const storedBlockchainStatus = localStorage.getItem("blockchainEnabled");

    if (storedBlockchainStatus) {
      setBlockchainEnabled(storedBlockchainStatus === "true");
    }

    if (token && username) {
      setAuth({
        isAuthenticated: true,
        token,
        username,
      });
    }
  }, []);

  // Connect to socket when authenticated
  useEffect(() => {
    if (!auth.isAuthenticated) return;

    // Initialize socket with auth token
    const newSocket = io(API_BASE_URL, {
      auth: {
        token: auth.token,
      },
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    newSocket.on("connect", () => {
      console.log("Connected to server");
      setSocketConnected(true);
      setSendError("");
    });

    newSocket.on("disconnect", () => {
      console.log("Disconnected from server");
      setSocketConnected(false);
    });

    newSocket.on("connect_error", (error) => {
      console.error("Socket connection error:", error);
      setSocketConnected(false);
      setSendError("Connection error. Please check your network connection.");
    });

    newSocket.on("blockchain-status", (data) => {
      console.log("Blockchain status:", data);
      setBlockchainEnabled(data.enabled);
      localStorage.setItem("blockchainEnabled", data.enabled.toString());
    });

    newSocket.on("error", (error) => {
      console.error("Socket error:", error);
      setSendError(error.message || "An error occurred");
      setIsMessageSending(false);
      
      // Handle authentication errors
      if (error.message === "Authentication error") {
        handleLogout();
      }
    });

    newSocket.on("receive-message", (message) => {
      // Update messages state to include the new message
      setMessages((prev) => {
        // Find if we have a temporary "sending" version of this message
        const messageIndex = prev.findIndex(
          msg => 
            msg.sending && 
            msg.username === auth.username && 
            msg.content === message.content
        );
        
        if (messageIndex !== -1) {
          // Replace the temporary message with the confirmed one from server
          const updatedMessages = [...prev];
          updatedMessages[messageIndex] = message;
          return updatedMessages;
        } else {
          // Just add the new message
          return [...prev, message];
        }
      });
      
      // Clear sending state if this was our message
      if (message.username === auth.username) {
        setIsMessageSending(false);
        setSendError("");
      }
      
      // Scroll to bottom when new message arrives
      setTimeout(scrollToBottom, 100);
    });

    // Listen for message deletion events
    newSocket.on("message-deleted", (data) => {
      setMessages((prev) => prev.filter(msg => msg.id !== data.id));
      setDeleteError("");
    });

    // Listen for delete success events
    newSocket.on("delete-success", (data) => {
      console.log(`Message ${data.id} successfully deleted`);
      setDeleteError("");
    });

    // Listen for delete error events
    newSocket.on("delete-error", (data) => {
      console.error("Error deleting message:", data.error);
      setDeleteError(data.error || "Failed to delete message");
    });

    setSocket(newSocket);

    // Cleanup on unmount
    return () => {
      newSocket.disconnect();
    };
  }, [auth.isAuthenticated]);

  // Fetch message history after authentication
  useEffect(() => {
    if (!auth.isAuthenticated) return;

    const fetchMessages = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/messages`, {
          headers: {
            Authorization: `Bearer ${auth.token}`,
          },
        });
        setMessages(response.data.messages || []);
        
        // Scroll to bottom after messages load
        setTimeout(scrollToBottom, 100);
      } catch (error) {
        console.error("Error fetching messages:", error);
        if (error.response?.status === 401 || error.response?.status === 403) {
          handleLogout();
        }
      }
    };

    fetchMessages();
  }, [auth.isAuthenticated]);

  // Effect to scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleLogin = (userData) => {
    setAuth({
      isAuthenticated: true,
      token: userData.token,
      username: userData.username,
    });
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    localStorage.removeItem("blockchainEnabled");
    setAuth({
      isAuthenticated: false,
      token: null,
      username: null,
    });
    setMessages([]);
    if (socket) {
      socket.disconnect();
    }
  };

  const addEmoji = (emoji) => {
    setInput(input + emoji);
    setShowEmojis(false);
  };

  const sendMessage = () => {
    if (!input.trim() || !socket || isMessageSending) return;

    const timestamp = Date.now();
    const messageData = {
      content: input.trim(),
      timestamp,
    };

    setIsMessageSending(true);
    setSendError("");
    socket.emit("send-message", messageData);
    setInput("");
    
    // Add temporary "sending" message to the UI immediately
    setMessages((prev) => [
      ...prev, 
      {
        content: input.trim(),
        username: auth.username,
        timestamp,
        sending: true,
        id: `temp-${timestamp}` // Temporary ID for this message
      }
    ]);
  };

  const deleteMessage = (messageId) => {
    if (!socket) return;
    setDeleteError("");
    
    // Emit delete message event
    socket.emit("delete-message", { id: messageId });
    
    // Optionally, update UI immediately to indicate deletion in progress
    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { ...msg, deleting: true } 
        : msg
    ));
  };

  if (!auth.isAuthenticated) {
    return <AuthForm onLogin={handleLogin} />;
  }

  return (
    <div style={{ padding: "20px", maxWidth: "800px", margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
        <h1>Blockchain Chat App</h1>
        <div>
          <span>Logged in as <strong>{auth.username}</strong></span>
          <button
            onClick={handleLogout}
            style={{
              marginLeft: "10px",
              padding: "5px 10px",
              backgroundColor: "#f44336",
              color: "white",
              border: "none",
              cursor: "pointer",
              borderRadius: "4px"
            }}
          >
            Logout
          </button>
        </div>
      </div>

      {!socketConnected && (
        <div style={{ 
          backgroundColor: "#ffebee", 
          color: "#c62828", 
          padding: "10px", 
          borderRadius: "4px",
          marginBottom: "15px" 
        }}>
          <strong>Warning:</strong> Not connected to server. Messages may not be sent or received.
        </div>
      )}

      <div
        style={{
          height: "500px",
          overflowY: "scroll",
          border: "1px solid #ddd",
          marginBottom: "15px",
          padding: "15px",
          borderRadius: "8px",
          backgroundColor: "#f9f9f9"
        }}
      >
        {messages.length === 0 ? (
          <p style={{ color: "#888", textAlign: "center" }}>
            No messages yet. Start a conversation!
          </p>
        ) : (
          messages.map((msg, index) => (
            <Message 
              key={msg.id || `msg-${index}`}
              msg={msg}
              currentUser={auth.username}
              onDelete={deleteMessage}
            />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {sendError && (
        <div style={{ color: "red", marginBottom: "10px" }}>
          Error: {sendError}
        </div>
      )}
      
      {deleteError && (
        <div style={{ color: "red", marginBottom: "10px" }}>
          Error deleting message: {deleteError}
        </div>
      )}

      <div style={{ display: "flex", position: "relative" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Type a message..."
          style={{
            flex: 1,
            padding: "12px",
            borderRadius: "8px",
            border: "1px solid #ddd",
            marginRight: "10px",
          }}
          disabled={isMessageSending || !socketConnected}
        />
        
        <button
          onClick={() => setShowEmojis(!showEmojis)}
          style={{
            backgroundColor: "#f1f1f1",
            border: "none",
            padding: "10px",
            borderRadius: "8px",
            cursor: "pointer",
            marginRight: "10px",
          }}
          disabled={!socketConnected}
        >
          😊
        </button>
        
        <button
          onClick={sendMessage}
          disabled={!input.trim() || isMessageSending || !socketConnected}
          style={{
            backgroundColor: "#4CAF50",
            color: "white",
            border: "none",
            padding: "10px 15px",
            borderRadius: "8px",
            cursor: (!input.trim() || isMessageSending || !socketConnected) ? "not-allowed" : "pointer",
            opacity: (!input.trim() || isMessageSending || !socketConnected) ? 0.7 : 1,
          }}
        >
          {isMessageSending ? "Sending..." : "Send"}
        </button>
        
        {showEmojis && (
          <div
            style={{
              position: "absolute",
              bottom: "50px",
              right: "0",
              backgroundColor: "white",
              border: "1px solid #ddd",
              borderRadius: "8px",
              padding: "10px",
              display: "flex",
              flexWrap: "wrap",
              width: "200px",
              boxShadow: "0 3px 10px rgba(0,0,0,0.2)",
            }}
          >
            {commonEmojis.map((emoji, index) => (
              <button
                key={index}
                onClick={() => addEmoji(emoji)}
                style={{
                  fontSize: "20px",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  padding: "5px",
                }}
              >
                {emoji}
              </button>
            ))}
          </div>
        )}
      </div>
      
      <div style={{ marginTop: "20px", textAlign: "center", fontSize: "0.8rem", color: "#888" }}>
        <p>
          {blockchainEnabled 
            ? "All messages are stored securely on a blockchain. Your messages are immutable and protected."
            : "Blockchain integration is currently disabled. Messages are stored in the database only."}
        </p>
      </div>
    </div>
  );
}

export default App;