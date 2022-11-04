window.onload=function(){
    document.querySelector(".btn-predict").onclick = send_data;

}

let loadFile = function(event) {
    const image = document.getElementById('output');
    image.src = URL.createObjectURL(event.target.files[0]);
};

let send_data = ()=>{
    console.log("Hello")
    let ajax = new XMLHttpRequest()
    let formData = new FormData(document.getElementById('predict_form'));
    //formData.append("image", file)
    ajax.open('POST', 'predict', true);
    //ajax.setRequestHeader("Content-Type", "multipart/form-data")
    ajax.onreadystatechange = ()=>{
        if(this.status === 200)
            console.log(this.responseText);
    }
    console.log(formData)
    ajax.send(formData)
}