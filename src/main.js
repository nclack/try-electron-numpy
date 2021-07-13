const { app, BrowserWindow } = require('electron')

function createWindow() {
    const win = new BrowserWindow({
        width: 800,
        height: 600
    })

    win.loadFile('./assets/index.html')
}

app.whenReady().then(()=>{
    createWindow()

    app.on('activate',()=>{
        if(BrowserWindow.getAllWindows().length === 0) {
            createWindow()
        }
    })
})