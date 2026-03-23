"use client";

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const API_URL = "https://baby-bloom-api-onbj.onrender.com/ask";
const HEALTH_URL = "https://baby-bloom-api-onbj.onrender.com/health";

export default function BabyBloomChat() {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    { role: 'bot', content: "Hi! I'm Baby Blooms. I'm here to answer your questions about newborn health based on clinical textbooks. How can I help today?" }
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const [chatHistory, setChatHistory] = useState<{role: string, content: string}[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  useEffect(() => {
    // Ping the server to wake it up from Render's free tier sleep
    fetch(HEALTH_URL).catch(() => {});
  }, []);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg = input.trim();
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setInput('');
    setIsTyping(true);

    try {
      // Set a generous timeout (300s) for Render's first deep cold start
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 300000);

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          question: userMsg,
          chat_history: chatHistory
        })
      });

      clearTimeout(timeoutId);
      const data = await response.json();
      setIsTyping(false);

      if (data.answer) {
        setMessages(prev => [...prev, { role: 'bot', content: data.answer }]);
        setChatHistory(prev => [
          ...prev, 
          { role: 'user', content: userMsg }, 
          { role: 'assistant', content: data.answer }
        ]);
      } else {
        setMessages(prev => [...prev, { role: 'bot', content: "I'm sorry, I'm having trouble connecting to my brain right now." }]);
      }
    } catch (error: any) {
      console.error("Chat Error:", error);
      setIsTyping(false);
      
      if (error.name === 'AbortError') {
        setMessages(prev => [...prev, { role: 'bot', content: "I'm still waking up my AI engine (this usually takes 30-50 seconds on the *first* request). Please wait a moment and try asking again! ✨" }]);
      } else {
        setMessages(prev => [...prev, { role: 'bot', content: "Error connecting to server. Please ensure the API is live." }]);
      }
    }
  };

  return (
    <div className="fixed bottom-8 right-8 z-[9999] font-sans">
      {/* Toggle Button */}
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={() => setIsOpen(!isOpen)}
        className="w-16 h-16 rounded-full bg-gradient-to-br from-[#ff8fa3] to-[#ff4d6d] shadow-2xl flex items-center justify-center cursor-pointer border-none text-3xl"
      >
        👶
      </motion.button>

      {/* Chat Window */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className="absolute bottom-20 right-0 w-[380px] h-[580px] bg-white/80 backdrop-blur-xl border border-white/20 rounded-[2rem] shadow-[0_20px_60px_rgba(0,0,0,0.15)] flex flex-col overflow-hidden"
          >
            {/* Header */}
            <div className="p-6 bg-gradient-to-r from-[#ff8fa3] to-[#ff4d6d] text-white flex items-center gap-4">
              <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center text-2xl shadow-inner">
                🩺
              </div>
              <div>
                <div className="font-bold text-lg leading-tight">Baby Blooms</div>
                <div className="text-xs opacity-90 flex items-center gap-1">
                  <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                  Online | AI Specialist
                </div>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 p-6 overflow-y-auto space-y-4 bg-gray-50/30">
              {messages.map((msg, idx) => (
                <motion.div
                  initial={{ opacity: 0, x: msg.role === 'user' ? 20 : -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  key={idx}
                  className={`max-w-[85%] p-4 rounded-2xl text-sm leading-relaxed shadow-sm ${
                    msg.role === 'user' 
                    ? 'ml-auto bg-[#ff4d6d] text-white rounded-br-none' 
                    : 'mr-auto bg-white text-gray-800 rounded-bl-none border border-gray-100'
                  }`}
                  dangerouslySetInnerHTML={{ 
                    __html: msg.content
                      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                      .replace(/\n/g, '<br/>') 
                  }}
                />
              ))}
              {isTyping && (
                <div className="mr-auto bg-white p-4 rounded-2xl rounded-bl-none border border-gray-100 flex gap-1 shadow-sm">
                  <span className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce" />
                  <span className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce [animation-delay:0.2s]" />
                  <span className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce [animation-delay:0.4s]" />
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-5 bg-white border-t border-gray-100 flex gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Ask about newborn health..."
                className="flex-1 bg-gray-100 border-none rounded-2xl px-5 py-3 text-sm focus:ring-2 focus:ring-[#ff8fa3] transition-all outline-none text-gray-700"
              />
              <button
                onClick={handleSend}
                className="bg-[#ff4d6d] text-white w-12 h-12 rounded-2xl flex items-center justify-center transition-all hover:bg-[#ff8fa3] active:scale-95 disabled:opacity-50"
                disabled={isTyping || !input.trim()}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13"></line>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
              </button>
            </div>
            
            <div className="px-6 py-2 bg-white text-[10px] text-gray-400 text-center border-t border-gray-50">
              AI assistance isn't a substitute for medical advice.
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
