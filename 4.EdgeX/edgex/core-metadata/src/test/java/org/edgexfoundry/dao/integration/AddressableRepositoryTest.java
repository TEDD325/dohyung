/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.dao.integration;

import static org.edgexfoundry.test.data.AddressableData.TEST_ADDRESS;
import static org.edgexfoundry.test.data.AddressableData.TEST_ADDR_NAME;
import static org.edgexfoundry.test.data.AddressableData.TEST_PORT;
import static org.edgexfoundry.test.data.AddressableData.TEST_PUBLISHER;
import static org.edgexfoundry.test.data.AddressableData.TEST_TOPIC;
import static org.edgexfoundry.test.data.AddressableData.checkTestData;
import static org.edgexfoundry.test.data.AddressableData.newTestInstance;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.dao.AddressableRepository;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class})
public class AddressableRepositoryTest {

  @Autowired
  private AddressableRepository repos;

  private String addressableId;

  /**
   * Create and save an instance of the Addressable before each test Note: the before method tests
   * the save operation of the Repository
   */
  @Before
  public void creatTestData() {
    Addressable addr = newTestInstance();
    repos.save(addr);
    addressableId = addr.getId();
    assertNotNull("new test Addressable has no identifier", addressableId);
  }

  @After
  public void cleanup() {
    repos.deleteAll();
  }

  @Test
  public void testFindOne() {
    Addressable addr = repos.findOne(addressableId);
    assertNotNull("Find one returns no addressable", addr);
    checkTestData(addr, addressableId);
  }

  @Test
  public void testFindOneWithBadId() {
    Addressable addr = repos.findOne("foo");
    assertNull("Find one returns addressable with bad id", addr);
  }

  @Test
  public void testFindAll() {
    List<Addressable> addrs = repos.findAll();
    assertEquals("Find all not returning a list with one addressable", 1, addrs.size());
    checkTestData(addrs.get(0), addressableId);
  }

  @Test
  public void testFindByName() {
    Addressable addr = repos.findByName(TEST_ADDR_NAME);
    assertNotNull("Find by name returns no addressable", addr);
    checkTestData(addr, addressableId);
  }

  @Test
  public void testFindByNameWithBadName() {
    Addressable addr = repos.findByName("badname");
    assertNull("Find by name returns addressable with bad name", addr);
  }

  @Test
  public void testFindByAddress() {
    List<Addressable> addrs = repos.findByAddress(TEST_ADDRESS);
    assertEquals("Find by address returns no addressable", 1, addrs.size());
    checkTestData(addrs.get(0), addressableId);
  }

  @Test
  public void testFindByAddressWithBadAddress() {
    List<Addressable> addrs = repos.findByAddress("badaddress");
    assertTrue("Find by address returns addressable with bad address", addrs.isEmpty());
  }

  @Test
  public void testFindByPort() {
    List<Addressable> addrs = repos.findByPort(TEST_PORT);
    assertEquals("Find by port returns no addressable", 1, addrs.size());
    checkTestData(addrs.get(0), addressableId);
  }

  @Test
  public void testFindByPortWithBadPort() {
    List<Addressable> addrs = repos.findByPort(8080);
    assertTrue("Find by port returns addressable with bad port", addrs.isEmpty());
  }

  @Test
  public void testFindByTopic() {
    List<Addressable> addrs = repos.findByTopic(TEST_TOPIC);
    assertEquals("Find by topic returns no addressable", 1, addrs.size());
    checkTestData(addrs.get(0), addressableId);
  }

  @Test
  public void testFindByTopicWithBadTopic() {
    List<Addressable> addrs = repos.findByTopic("badtopic");
    assertTrue("Find by topic returns addressable with bad topic", addrs.isEmpty());
  }

  @Test
  public void testFindByPublisher() {
    List<Addressable> addrs = repos.findByPublisher(TEST_PUBLISHER);
    assertEquals("Find by publisher returns no addressable", 1, addrs.size());
    checkTestData(addrs.get(0), addressableId);
  }

  @Test
  public void testFindByTopicWithBadPublisher() {
    List<Addressable> addrs = repos.findByPublisher("badpublisher");
    assertTrue("Find by publisher returns addressable with bad publisher", addrs.isEmpty());
  }

  @Test(expected = DuplicateKeyException.class)
  public void testAddAddressWithSameName() {
    Addressable addr = newTestInstance();
    repos.save(addr);
    fail("Should not have been able to save the address with a duplicate name");
  }

  @Test
  public void testUpdate() {
    Addressable addr = repos.findOne(addressableId);
    // check that create and modified timestamps are the same
    assertEquals("Modified and created timestamps should be equal after creation", addr.getModified(),
        addr.getCreated());
    addr.setPassword("newpass");
    repos.save(addr);
    // reread addressable
    Addressable addr2 = repos.findOne(addressableId);
    assertEquals("Addressable was not updated appropriately", "newpass", addr2.getPassword());
    assertNotEquals(
        "after modification, modified timestamp still the same as the addressable's create timestamp",
        addr2.getModified(), addr2.getCreated());
  }

  @Test
  public void testDelete() {
    Addressable addr = repos.findOne(addressableId);
    repos.delete(addr);
    assertNull("Addressable not deleted", repos.findOne(addressableId));
  }

}
