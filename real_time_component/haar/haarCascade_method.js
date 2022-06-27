let add_button = document.getElementById("add");
let zoom_window_text = document.getElementById("zoom_window_text");
let process_window_text = document.getElementById("process_window_text");
let video = document.getElementById("cam_input");
let output = document.getElementById("canvas_output");
let container = document.getElementById("container");
let utils = new Utils();

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
})


function openCvReady() {
    // console.log(cv);
    let FPS = 30;
    let video = document.getElementById("cam_input");
    console.log('creat frame:', video.width, video.height);
    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let gray = new cv.Mat();
    let cap = new cv.VideoCapture(cam_input);
    let faces = new cv.RectVector();
    let classifier = new cv.CascadeClassifier();
    let minsize = new cv.Size(0, 0);
    let maxsize = new cv.Size(1000, 1000);
    let faceCascadeFile = 'haarcascade_frontalface_default.xml';
    let faceCascadeFileLink = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    utils.createFileFromUrl(faceCascadeFile, faceCascadeFileLink, () => {
        document.getElementById('middle').innerHTML = 'Downloading haar-cascade model';
        try{classifier.load(faceCascadeFile);} // in the callback, load the cascade from file 
        catch(err){console.log(err);}
        document.getElementById('middle').innerHTML = '';
    });
    let face_row = -1;
    let face_col = -1;
    let clip_width = video.width/5;
    let clip_height = video.height/5;
    function processVideo() {
        let begin = Date.now();
        if (video.srcObject!=null){
            cap.read(src);
            src.copyTo(dst);
            cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
            try{
                classifier.detectMultiScale(gray, faces, 1.1, 3);
                let today = new Date();
                let time = today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds();
                console.log("time: ", time, " face size: "+ faces.size());
            }catch(err){
                console.log(err);
            }for (let i = 0; i < faces.size(); ++i) {
                let face = faces.get(i);
                console.log('face (' + i + ') ' + [face.x, face.y, face.width, face.height]);
                let face_row = parseInt(face.y/clip_height);
                let face_col = parseInt(face.x/clip_width);

                let tmp_row = parseInt((face.y+face.height-5) / clip_height);
                let tmp_col = parseInt((face.x+face.width-5) / clip_width);
                // console.log([face_row, face_col, tmp_row, tmp_col]);
                //check for error
                // if (face.width>=clip_width || face.height>=clip_height || tmp_row!=face_row || tmp_col!=face_col)
                //     continue;

                let point1 = new cv.Point(face.x, face.y);
                let point2 = new cv.Point(face.x + face.width, face.y + face.height);
                cv.rectangle(dst, point1, point2, [73, 171, 232, 255], 2);
            }
            cv.imshow("canvas_output", dst);
        }
        // schedule next one.
        let delay = 1000/FPS - (Date.now() - begin);
        setTimeout(processVideo, delay);
    }
    // schedule first one.
    setTimeout(processVideo, 0); 
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