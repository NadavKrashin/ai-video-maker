import React from 'react';
import ReactDOM from 'react-dom/client';
import '@mantine/core/styles.css';
import '@mantine/notifications/styles.css';
import { MantineProvider, createTheme } from '@mantine/core';
import { Notifications } from '@mantine/notifications';
import App from './App.jsx';

const theme = createTheme({
  primaryColor: 'orange',
  defaultRadius: 'md',
  cursorType: 'pointer'
});

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <MantineProvider theme={theme} defaultColorScheme="dark">
      <Notifications position="bottom-center" autoClose={5000} />
      <App />
    </MantineProvider>
  </React.StrictMode>
);
