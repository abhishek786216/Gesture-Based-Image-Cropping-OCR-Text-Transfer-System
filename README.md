# ✋ Gesture Transfer

Seamless **gesture-based file and message transfer** over the same **local network (LAN)**.  
This project combines **Human-Computer Interaction (HCI)** with **socket programming** to enable contactless sharing using **hand gestures**.  

---

## ✨ Overview
**Gesture Transfer** is a prototype that allows you to send files or messages to another device on the same LAN using **hand gestures as triggers**.  
Built with **Python, OpenCV, Mediapipe, and Sockets**, it demonstrates how gesture recognition can be integrated with networking for a unique, contactless experience.  

---

## 🛠 Tech Stack
- 🐍 **Python 3.x** – Core language  
- 👁 **OpenCV** – Computer vision  
- 🖐 **Mediapipe** – Hand tracking & gesture recognition  
- 🌐 **Socket Programming** – LAN communication  

---

## ⚡ Features
- ✋ Transfer triggered by gestures (no clicks needed)  
- 📡 Real-time file/message transfer across LAN  
- 🔒 Configurable IP & Port (works on any local device)  
- 💻 Cross-platform prototype (Windows/Linux tested)    

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Cyber-Hades/GestureTransfer
cd GestureTransfer
```
---

### 2️⃣ Configure IP & Port
```bash
Update the config.json file:
{
  "this_device_ip": "192.xxx.xxx.xxx",
  "peer_device_ip": "192.xxx.xxx.xxx",
  "port": 9999
}
```
- **this_device_ip** → Your device’s LAN IP  
- **peer_device_ip** → Peer device’s LAN IP  
- **port** → Any free port (default: `9999`)  

💡 **Find your IP** using `ipconfig` (Windows) or `ifconfig` (Linux/Mac)  
---
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
---
### 4️⃣ Run the project
```bash
python main.py
```
---
## 🎯 Use Cases

- 📂 **Contactless file sharing** in labs, classrooms, or offices  
- 🖐 **Exploring Human-Computer Interaction (HCI)** concepts  
- 🎓 **Academic projects** combining networking & gesture recognition  


