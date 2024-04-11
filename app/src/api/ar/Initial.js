import React from 'react';
import { instanceOf } from 'prop-types';
import { withCookies, Cookies } from 'react-cookie';
import '../../App.css';

// Conducts the initial ratings - button submits rating and loads next
class Initial extends React.Component {

  static propTypes = {
    cookies: instanceOf(Cookies).isRequired
  };

  constructor(props) {
    super(props);
    const { cookies } = props;
    this.state = {
      img: "",
      user: cookies.get('user'),
      img_fallback: "https://cdnjs.cloudflare.com/ajax/libs/galleriffic/2.0.1/css/loader.gif"
    }
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  // React function that is called when the page loads.
  async componentDidMount() {
    // Send an HTTP request to the server.

    fetch(
      process.env.REACT_APP_BACKEND_ENDPOINT + '/initial_send/' + this.state.user, { method: 'GET' } // The type of HTTP request.
    ).then(
      res => {
        // Convert the response data to a JSON.
        return res.json();
      }, err => {
        // Print the error if there is one.
        console.log(err);
      }
    ).then(
      data => {
        // Act on data
        if (data.move_on === true) {
          // Move on to refining stage
          this.nextStage();
        }
        else {
          this.setState({
            // Load next image
            img: data.img_url
          });
        }
      }, err => {
        // Print the error if there is one.
        console.log(err);
      }
    );
  }

  timeout(delay) {
    return new Promise(res => setTimeout(res, delay));
  }

  // Finish the initial stage
  async nextStage() {
    // Call ahead to build the model
    try {
      await fetch(process.env.REACT_APP_BACKEND_ENDPOINT + '/build_model/' + this.state.user, { method: 'POST', body: "", });
      window.location.href = "/refine";
    } catch (error) {
      console.log(error);
    }
  }

  // Submit the rating to model and refresh
  async handleSubmit(event) {
    event.preventDefault();
    const data = new FormData(event.target);

    // Unload current image and rating
    document.getElementById('rating').value = "";
    this.setState({
      img: ""
    });

    // Send rating to model and await calculation
    fetch(process.env.REACT_APP_BACKEND_ENDPOINT + '/initial_receive/' + this.state.user, {
      method: 'POST',
      body: data,
    }).then(
      res => { this.componentDidMount(); },
      err => { console.log(err) }
    );
  }

  render() {
    let page_body;

    if (this.state.img === "") {
      page_body = (
        <div>
          <h2>Loading</h2>
          <img src={this.state.img_fallback} className="Loading" alt="Loading next dress" />
        </div>
      )
    } else {
      page_body = (
        <div>
          <p>Please rate the following dress on a scale from 0 to 10 (higher is better):</p>
          <img src={this.state.img} className="App-photo" alt="Please rate this dress." />
          <br />
          <form onSubmit={this.handleSubmit}>
            <label htmlFor="rating">Enter your rating for this dress</label>
            <br />
            <input id="rating" name="rating" type="number" step="0.1" min="0" max="10" required />
            <button className="App-button">Submit</button>
            <br />
          </form>
        </div>
      )
    }
    return (
      <div className="App">
        <header className="App-header">
          {page_body}
        </header>
      </div>
    );
  }
}

export default withCookies(Initial);