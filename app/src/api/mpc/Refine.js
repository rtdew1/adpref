import React from 'react';
import { instanceOf } from 'prop-types';
import { withCookies, Cookies } from 'react-cookie';
import '../../App.css';

// Conducts the refining ratings - button submits rating and loads next
class Refine extends React.Component {

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
      'https://s7d2.scene7.com/is/image/nu/4130059680055_049_b?$pdp-regular$',
      'https://s7d2.scene7.com/is/image/nu/4130024090046_095_b?$pdp-regular$',
      'https://s7d2.scene7.com/is/image/nu/4130077370111_049_b?$pdp-regular$',
      'https://s7d2.scene7.com/is/image/nu/36521532_012_b?$pdp-regular$'
    ]
  }

  // React function that is called when the page loads.
  componentDidMount() {
    // Send an HTTP request to the server.
    fetch(
      process.env.REACT_APP_BACKEND_ENDPOINT + '/refine_send/' + this.state.user, { method: 'GET' } // The type of HTTP request.
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
          // Move on to testing stage
          this.nextStage()
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

  timeout(delay) {
    return new Promise(res => setTimeout(res, delay));
  }

  // Finish the refining stage
  async nextStage() {
    // Call ahead to calculate testing items
    try {
      await fetch(process.env.REACT_APP_BACKEND_ENDPOINT + '/calc_results/' + this.state.user, {
        method: 'POST',
        body: "",
      })

      const res = (await fetch(
        process.env.REACT_APP_BACKEND_ENDPOINT + '/num_testing/' + this.state.user, { method: 'GET' }
      )).json()

      this.setState({
        num_testing: res.num_testing
      })

      window.location.href = "/test"
    } catch (error) {
      console.log(error)
    }
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

    // Send rating to model and await calculation
    fetch(process.env.REACT_APP_BACKEND_ENDPOINT + '/refine_receive/' + this.state.user, {
      method: 'POST',
      body: data,
    }).then(
      res => { this.componentDidMount(); },
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

export default withCookies(Refine);
