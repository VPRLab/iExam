let add_button = document.getElementById("add");
let zoom_window_text = document.getElementById("zoom_window_text");
let process_window_text = document.getElementById("process_window_text");
let video = document.getElementById("cam_input");
let output = document.getElementById("canvas_output");
let container = document.getElementById("container");
let utils = new Utils();
var netDet = undefined;
var persons = {};

function checkBrowser() {
    if((navigator.userAgent.indexOf("Opera") || navigator.userAgent.indexOf('OPR')) != -1 || navigator.userAgent.indexOf("Chrome") != -1 || navigator.userAgent.indexOf("Firefox") != -1 || ((navigator.userAgent.indexOf("MSIE") != -1 ) || (!!document.documentMode == true )))
        console.log("browser owns capture window function")
    else if(navigator.userAgent.indexOf("Safari") != -1){
        while(true)
            alert("You use Safari now, please change another browser because Safari not support capture window function");
    }
    else 
       alert('unknown');
}

function btn1_click() {
    let today = new Date();
    let time = today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds();
    console.log('start exam ' + time);
}

function btn2_click() {
    let today = new Date();
    let time = today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds();
    console.log('end up exam ' + time);
}

add_button.addEventListener("click", function(e) {
    navigator.mediaDevices.getDisplayMedia({ video: true, audio: false })
    .then(function(stream) {
        let settings = stream.getVideoTracks()[0].getSettings();
        console.log(settings);
        add_button.style.display = "none";
        zoom_window_text.style.display = "none";
        process_window_text.style.display = "none";
        console.log('dimension:', video.width, video.height);
        console.log(window.innerWidth, 'output_width:', output_width, video.width);
            
        // FPS = settings.frameRate;
        var output_width = window.innerWidth-300;
        output.width = output_width;
        video.width = output_width;
        video.srcObject = stream;
        video.play();
        console.log('dimension:', video.width, video.height);
        console.log(window.innerWidth, 'output_width:', output_width, video.width);
         
    })
    .then(()=> {
        utils.loadOpenCv(openCvReady);
    })
    .catch(function(err) {
        console.log("An error occurred! " + err);
    });
});

function loadModels(callback) {
    var proto = 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt';
    var weights = 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel';
    utils.createFileFromUrl('face_detector.prototxt', proto, () => {
      document.getElementById('middle').innerHTML = 'Downloading face_detector.caffemodel';
      utils.createFileFromUrl('face_detector.caffemodel', weights, () => {
          document.getElementById('middle').innerHTML = 'Downloading OpenFace model';
          document.getElementById('middle').innerHTML = '';
          netDet = cv.readNetFromCaffe('face_detector.prototxt', 'face_detector.caffemodel');
          callback();
      });
    });
  };

function openCvReady() {
    // console.log(cv);
    let FPS = 30;
    let video = document.getElementById("cam_input");
    // console.log('creat frame:', video.width, video.height);
    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let frameBGR = new cv.Mat(video.height, video.width, cv.CV_8UC3);
    let gray = new cv.Mat();
    let cap = new cv.VideoCapture(cam_input);
    function processVideo() {
        let begin = Date.now();
        if (video.srcObject!=null){
            cap.read(src);
            src.copyTo(dst);
            cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
            try{
                cv.cvtColor(src, frameBGR, cv.COLOR_RGBA2BGR);

                var faces = detectFaces(frameBGR);
                let today = new Date();
                let time = today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds();
                console.log("time: ", time, " face size: "+ faces.length);
                faces.forEach(function(rect) {
                    console.log('x:', rect.x, 'y:', rect.y, 'width:', rect.width, 'height:', rect.height);
                });
                
                faces.forEach(function(rect) {
                    cv.rectangle(dst, {x: rect.x, y: rect.y}, {x: rect.x + rect.width, y: rect.y + rect.height}, [0, 255, 0, 255]);
                });
            }catch(err){
                console.log(err);
            }
            cv.imshow("canvas_output", dst);
        }
        // schedule next one.
        let delay = 1000/FPS - (Date.now() - begin);
        setTimeout(processVideo, delay);
    }
    function detectFaces(img) {
        var blob = cv.blobFromImage(img, 1, {width: 300, height: 300});
        netDet.setInput(blob);
        var out = netDet.forward();      
        var faces = [];
        for (var i = 0, n = out.data32F.length; i < n; i += 7) {
          var confidence = out.data32F[i + 2];
          var left = out.data32F[i + 3] * img.cols;
          var top = out.data32F[i + 4] * img.rows;
          var right = out.data32F[i + 5] * img.cols;
          var bottom = out.data32F[i + 6] * img.rows;
          left = Math.min(Math.max(0, left), img.cols - 1);
          right = Math.min(Math.max(0, right), img.cols - 1);
          bottom = Math.min(Math.max(0, bottom), img.rows - 1);
          top = Math.min(Math.max(0, top), img.rows - 1);
      
          if (confidence > 0.5 && left < right && top < bottom) {
            faces.push({x: left, y: top, width: right - left, height: bottom - top})
          }
        }
        blob.delete();
        out.delete();
        return faces;
    };
    // schedule first one.
    loadModels(processVideo);
}



function Utils() {
    let self = this;
    this.createFileFromUrl = function(path, url, callback) {
        let request = new XMLHttpRequest();
        request.open('GET', url, true);
        request.responseType = 'arraybuffer';
        request.onload = function(ev) {
            if (request.readyState === 4) {
                if (request.status === 200) {
                    let data = new Uint8Array(request.response);
                    cv.FS_createDataFile('/', path, data, true, false, false);
                    callback();
                } else {
                    self.printError('Failed to load ' + url + ' status: ' + request.status);
                }
            }
        };
        request.send();
    };
    const OPENCV_URL = 'opencv.js';
    this.loadOpenCv = function(onloadCallback) {
        let script = document.createElement('script');
        script.setAttribute('async', '');
        script.setAttribute('type', 'text/javascript');
        script.addEventListener('load', async () => {
            if (cv.getBuildInformation)
            {
                console.log(cv.getBuildInformation());
                onloadCallback();
            }
            else
            {
                // WASM
                if (cv instanceof Promise) {
                    cv = await cv;
                    console.log(cv.getBuildInformation());
                    onloadCallback();
                } else {
                    cv['onRuntimeInitialized']=()=>{  //satisfy this condition
                        console.log(cv.getBuildInformation()); 
                        onloadCallback();
                    }
                }
            }
        });
        script.addEventListener('error', () => {
            self.printError('Failed to load ' + OPENCV_URL);
        });
        script.src = OPENCV_URL;
        let node = document.getElementsByTagName('script')[0];
        node.parentNode.insertBefore(script, node);
    };
}