import React from 'react';
import { instanceOf } from 'prop-types';
import { withCookies, Cookies } from 'react-cookie';
import '../../App.css';

// Conducts the testing ratings - button submits rating and loads next
class Test extends React.Component {

  static propTypes = {
    cookies: instanceOf(Cookies).isRequired
  };

  constructor(props) {
    super(props);
    const { cookies } = props;
    this.state = {
      img_1: "",
      img_2: "",
      user: cookies.get('user'),
      img_fallback: "https://cdnjs.cloudflare.com/ajax/libs/galleriffic/2.0.1/css/loader.gif",
      arbitrary_check: false,
      arbitrary_instruction: 0,
      arbitrary_count: 0
    }
    this.handleSubmit = this.handleSubmit.bind(this);
    this.arbitraryImageURLs = [
      'https://s7d2.scene7.com/is/image/nu/4130077370105_041_b?$pdp-regular$',
      'https://s7d2.scene7.com/is/image/nu/4130077370103_266_b?$pdp-regular$',
      'https://s7d2.scene7.com/is/image/nu/4130084320015_049_b?$pdp-regular$',
      'https://s7d2.scene7.com/is/image/nu/4130089540016_009_b?$pdp-regular$'
    ]
  }

  // React function that is called when the page loads.
  componentDidMount() {
    // Send an HTTP request to the server.
    fetch(
      process.env.REACT_APP_BACKEND_ENDPOINT + '/test_send/' + this.state.user, { method: 'GET' } // The type of HTTP request.
    ).then(
      res => {
        // Convert the response data to a JSON.
        return res.json();
      }, err => {
        // Print the error if there is one.
        console.log(err);
      }
    ).then
      (data => {
        // Act on data
        if (data.move_on === true) {
          // Move on to recommendation stage
          window.location.href = "/end_test"
        }
        else {
          this.setState({
            // Load next image
            img_1: data.img_url_1,
            img_2: data.img_url_2,
            arbitrary_check: data.arbitrary_check,
            arbitrary_instruction: data.arbitrary_instruction,
            arbitrary_count: this.state.arbitrary_count + (data.arbitrary_check ? 1 : 0)
          });

          if (this.state.arbitrary_check) {
            this.setState({
              img_1: this.getArbitraryImageURLLeft(),
              img_2: this.getArbitraryImageURLRight()
            })
          }
        }
      }, err => {
        // Print the error if there is one.
        console.log(err);
      }
      );
  }

  // Submit the rating to model and refresh
  async handleSubmit(event) {
    event.preventDefault();
    const data = new FormData(event.target);

    // Unload current image and rating
    document.getElementById('rating').value = "";
    this.setState({
      img_1: "",
      img_2: "",
    });

    // Send rating to model and await completion
    fetch(process.env.REACT_APP_BACKEND_ENDPOINT + '/test_receive/' + this.state.user, {
      method: 'POST',
      body: data,
    }).then(
      data => { this.componentDidMount() },
      err => { console.log(err) }
    );
  }

  getInstructionText(target_rating) {
    let m = new Map();
    m.set(-2, "strongly prefer left")
    m.set(-1, "prefer left")
    m.set(0, "neutral")
    m.set(1, "prefer right")
    m.set(2, "strongly prefer right")

    return m.get(parseInt(target_rating))
  }

  getArbitraryImageURLLeft() {
    return this.arbitraryImageURLs[
      (this.state.arbitrary_count - 1) % this.arbitraryImageURLs.length
    ]
  }

  getArbitraryImageURLRight() {
    return this.arbitraryImageURLs[
      this.state.arbitrary_count % this.arbitraryImageURLs.length
    ]
  }

  render() {
    let page_body;
    if ((this.state.img_1 === "") || (this.state.img_2 === "")) {
      page_body = (
        <div>
          <h2>Loading</h2>
          <img src={this.state.img_fallback} className="Loading" alt="Loading next dress" />
        </div>
      )
    } else {
      page_body = (
        <div>
          <p>Which of these two dresses would you prefer?:</p>
          <div className="img-container">
            <img src={this.state.img_1} className="img-left" alt="Please rate this dress." />
            &nbsp; or &nbsp;
            <img src={this.state.img_2} className="img-right" alt="Please rate this dress." />
          </div>
          <br />
          <form onSubmit={this.handleSubmit}>
            <label htmlFor="rating">
              {
                (this.state.arbitrary_check ? "Please choose " + this.getInstructionText(this.state.arbitrary_instruction) : "Enter your preference for the dresses")
              }
            </label>
            <br />
            <table className="likert"> 
                  <thead>
                    <tr>
                      <th> Strongly Prefer Left </th>
                      <th> Prefer left </th>
                      <th> Indifferent  </th>
                      <th> Prefer Right </th>
                      <th> Strongly Prefer Right </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td> <input type="radio" id="rating" name="rating" value="-2" required /> </td>
                      <td> <input type="radio" id="rating" name="rating" value="-1" required /> </td>
                      <td> <input type="radio" id="rating" name="rating" value="0" required /> </td>
                      <td> <input type="radio" id="rating" name="rating" value="1" required /> </td>
                      <td> <input type="radio" id="rating" name="rating" value="2" required /> </td>
                    </tr>
                  </tbody>
                </table>
            <button className="mpc-submit-button">Submit</button>
            <br />
          </form>
        </div>)
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

export default withCookies(Test);
