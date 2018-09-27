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

package org.edgexfoundry.controller.integration;

import static org.edgexfoundry.test.data.AddressableData.TEST_ADDRESS;
import static org.edgexfoundry.test.data.AddressableData.TEST_ADDR_NAME;
import static org.edgexfoundry.test.data.AddressableData.TEST_PORT;
import static org.edgexfoundry.test.data.AddressableData.TEST_PUBLISHER;
import static org.edgexfoundry.test.data.AddressableData.TEST_TOPIC;
import static org.edgexfoundry.test.data.AddressableData.checkTestData;
import static org.edgexfoundry.test.data.AddressableData.newTestInstance;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.lang.reflect.Field;
import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.controller.impl.AddressableControllerImpl;
import org.edgexfoundry.dao.AddressableRepository;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.DeviceServiceRepository;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.edgexfoundry.test.data.DeviceData;
import org.edgexfoundry.test.data.ServiceData;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class})
public class AddressableControllerTest {

  private static final String LIMIT = "maxLimit";

  @Autowired
  AddressableRepository repos;

  @Autowired
  DeviceRepository deviceRepos;

  @Autowired
  DeviceServiceRepository serviceRepos;

  @Autowired
  AddressableControllerImpl controller;

  private String id;

  @Before
  public void setup() {
    Addressable addr = newTestInstance();
    repos.save(addr);
    id = addr.getId();
  }

  @After
  public void cleanup() throws Exception {
    resetControllerMAXLIMIT();
    resetRepos();
    repos.deleteAll();
    deviceRepos.deleteAll();
    serviceRepos.deleteAll();
  }

  @Test
  public void testAddressable() {
    Addressable addr = controller.addressable(id);
    checkTestData(addr, id);
  }

  @Test(expected = NotFoundException.class)
  public void testAddressableWithUnknownnId() {
    controller.addressable("nosuchid");
  }

  @Test(expected = ServiceException.class)
  public void testAddressableException() throws Exception {
    unsetRepos();
    controller.addressable(id);
  }

  @Test
  public void testAddressables() {
    List<Addressable> as = controller.addressables();
    assertEquals("Find all not returning a list with one addressable", 1, as.size());
    checkTestData(as.get(0), id);
  }

  @Test(expected = ServiceException.class)
  public void testAddressablesException() throws Exception {
    unsetRepos();
    controller.addressables();
  }

  @Test(expected = LimitExceededException.class)
  public void testAddressablesMaxLimitExceeded() throws Exception {
    unsetControllerMAXLIMIT();
    controller.addressables();
  }

  @Test
  public void testAddressableForName() {
    Addressable addr = controller.addressableForName(TEST_ADDR_NAME);
    checkTestData(addr, id);
  }

  @Test(expected = NotFoundException.class)
  public void testAddressableForNameWithNoneMatching() {
    controller.addressableForName("badname");
  }

  @Test(expected = ServiceException.class)
  public void testAddressableForNameException() throws Exception {
    unsetRepos();
    controller.addressableForName(TEST_ADDR_NAME);
  }

  @Test
  public void testAddressablesForAddress() {
    List<Addressable> addrs = controller.addressablesByAddress(TEST_ADDRESS);
    assertEquals("Find for address not returning appropriate list", 1, addrs.size());
    checkTestData(addrs.get(0), id);
  }

  @Test(expected = ServiceException.class)
  public void testAddressablesForAddressWithNoneMatching() throws Exception {
    unsetRepos();
    assertTrue("No addressables should be found with bad address",
        controller.addressablesByAddress("badaddress").isEmpty());
  }

  @Test(expected = ServiceException.class)
  public void testAddressablesForAddressException() throws Exception {
    unsetRepos();
    controller.addressablesByAddress(TEST_ADDRESS);
  }

  @Test
  public void testAddressablesForPort() {
    List<Addressable> addrs = controller.addressablesByPort(TEST_PORT);
    assertEquals("Find for port not returning appropriate list", 1, addrs.size());
    checkTestData(addrs.get(0), id);
  }

  @Test(expected = ServiceException.class)
  public void testAddressablesForPortWithNoneMatching() throws Exception {
    unsetRepos();
    assertTrue("No addressables should be found with bad port",
        controller.addressablesByPort(1).isEmpty());
  }

  @Test(expected = ServiceException.class)
  public void testAddressablesForPortException() throws Exception {
    unsetRepos();
    controller.addressablesByPort(TEST_PORT);
  }

  @Test
  public void testAddressablesForTopic() {
    List<Addressable> as = controller.addressablesByTopic(TEST_TOPIC);
    assertEquals("Find for topic not returning appropriate list", 1, as.size());
    checkTestData(as.get(0), id);
  }

  @Test(expected = ServiceException.class)
  public void testAddressablesForTopicWithNoneMatching() throws Exception {
    unsetRepos();
    assertTrue("No addressables should be found with bad topic",
        controller.addressablesByTopic("badtopic").isEmpty());
  }

  @Test(expected = ServiceException.class)
  public void testAddressablesForTopicException() throws Exception {
    unsetRepos();
    controller.addressablesByTopic(TEST_TOPIC);
  }

  @Test
  public void testAddressablesForPublisher() {
    List<Addressable> addrs = controller.addressablesByPublisher(TEST_PUBLISHER);
    assertEquals("Find for publisher  not returning appropriate list", 1, addrs.size());
    checkTestData(addrs.get(0), id);
  }

  @Test(expected = ServiceException.class)
  public void testAddressablesForPublisherWithNoneMatching() throws Exception {
    unsetRepos();
    assertTrue("No addressables should be found with bad publisher",
        controller.addressablesByPublisher("badpublisher").isEmpty());
  }

  @Test(expected = ServiceException.class)
  public void testAddressablesForPublisherException() throws Exception {
    unsetRepos();
    controller.addressablesByPublisher(TEST_PUBLISHER);
  }

  @Test
  public void testAdd() {
    Addressable addr = newTestInstance();
    addr.setName("NewName");
    String newId = controller.add(addr);
    assertNotNull("New addressable id is null", newId);
    assertNotNull("Modified date is null", addr.getModified());
    assertNotNull("Create date is null", addr.getCreated());
  }

  @Test(expected = ServiceException.class)
  public void testAddNull() {
    controller.add(null);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithSameName() {
    Addressable addr = newTestInstance();
    controller.add(addr);
  }

  @Test(expected = ServiceException.class)
  public void testAddException() throws Exception {
    unsetRepos();
    Addressable addr = newTestInstance();
    addr.setName("NewName");
    controller.add(addr);
  }

  @Test
  public void testDelete() {
    assertTrue("Delete did not return correctly", controller.delete(id));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteWithNone() {
    controller.delete("badid");
  }

  @Test(expected = ServiceException.class)
  public void testDeleteException() throws Exception {
    unsetRepos();
    controller.delete(id);
  }

  @Test
  public void testDeleteByName() {
    assertTrue("Delete did not return correctly", controller.deleteByName(TEST_ADDR_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteByNameWithNone() {
    controller.delete("badname");
  }

  @Test(expected = ServiceException.class)
  public void testDeleteByNameException() throws Exception {
    unsetRepos();
    controller.deleteByName(TEST_ADDR_NAME);
  }

  @Test(expected = DataValidationException.class)
  public void testDeleteAssociatedToDevice() {
    Device device = DeviceData.newTestInstance();
    device.setAddressable(repos.findOne(id));
    deviceRepos.save(device);
    controller.delete(id);
  }

  @Test(expected = DataValidationException.class)
  public void testDeleteAssociatedToDeviceService() {
    DeviceService serv = ServiceData.newTestInstance();
    serv.setAddressable(repos.findOne(id));
    serviceRepos.save(serv);
    controller.delete(id);
  }

  @Test
  public void testUpdate() {
    Addressable addr = repos.findOne(id);
    addr.setAddress("newaddress");
    assertTrue("Update did not complete successfully", controller.update(addr));
    Addressable addr2 = repos.findOne(id);
    assertEquals("Update did not work correclty", "newaddress", addr2.getAddress());
    assertNotNull("Modified date is null", addr2.getModified());
    assertNotNull("Create date is null", addr2.getCreated());
    assertTrue("Modified date and create date should be different after update",
        addr2.getModified() != addr2.getCreated());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateException() throws Exception {
    unsetRepos();
    Addressable addr = repos.findOne(id);
    addr.setAddress("newaddress");
    controller.update(addr);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithNone() {
    Addressable addr = repos.findOne(id);
    addr.setId("badid");
    addr.setName("badname");
    addr.setAddress("newaddress");
    controller.update(addr);
  }

  private void unsetRepos() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField("repos");
    temp.setAccessible(true);
    temp.set(controller, null);
  }

  private void resetRepos() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField("repos");
    temp.setAccessible(true);
    temp.set(controller, repos);
  }

  private void unsetControllerMAXLIMIT() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT);
    temp.setAccessible(true);
    temp.set(controller, 0);
  }

  private void resetControllerMAXLIMIT() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT);
    temp.setAccessible(true);
    temp.set(controller, 1000);
  }
}
