from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/simulate', methods=['GET'])
def simulate():
    response = {
        "status": "success",
        "message": "Matriz de matrices",
        "data": "Placeholder para regresar la matriz de matrices"
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
