const {
  doSomethingUseful,
  onePythonCall,
} = require("../build/Release/iffy.node");
const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("iffy", {
  onePythonCall: onePythonCall,
});

window.addEventListener("DOMContentLoaded", () => {
  const replaceText = (selector, text) => {
    const el = document.getElementById(selector);
    if (el) el.innerText = text;
  };

  for (const dep of ["chrome", "node", "electron"]) {
    replaceText(`${dep}-version`, process.versions[dep]);
  }

  replaceText("hello-world", doSomethingUseful());

  //  console.log("before python call");
  //  console.log(onePythonCall("data.f64"));
  //  console.log("after python call");
});
