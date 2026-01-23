// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";

// Your web app's Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyBGKghmmuSL1sPoy0sBNLfh8WzpSZnEjoM",
    authDomain: "genbi-3ac9e.firebaseapp.com",
    projectId: "genbi-3ac9e",
    storageBucket: "genbi-3ac9e.firebasestorage.app",
    messagingSenderId: "381689575301",
    appId: "1:381689575301:web:e2e15735070a120ce276ec",
    measurementId: "G-5G23MN7R75"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

export { app, analytics };
